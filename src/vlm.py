"""
Grounded vision-language model. Visual tokens from the frozen encoder (encoder.py) -> MLP projector -> instruction-tuned
language model with LoRA (default Qwen2.5-1.5B-Instruct; any Hugging Face causal language model with a chat template
can be passed with --llm).

Task (one model per fold, same folds as the outcome models):
  masked scans     -> generate the findings report of make_reports.py
  labeled patients -> also give "OUTCOME: good|poor" (discharge mRS 0-2 vs 3-6), with the clinical sentence in the prompt
Outputs: results/vlm/gen_fold{k}.csv  (generated text, P(poor) from the next-token probabilities after "OUTCOME:")
Usage:  python src/vlm.py [--folds 0 1 2 3 4] [--llm Qwen/Qwen2.5-1.5B-Instruct] [--quick]
"""
import argparse, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import RES, WORK

SYSTEM = "You are a neuroradiology assistant. Describe acute ischemic stroke findings on diffusion MRI precisely."
INSTR_REP = "Describe the acute infarct."
INSTR_OUT = ("Describe the acute infarct and predict the discharge functional outcome "
             "(good = mRS 0-2, poor = mRS 3-6).")
INSTR_OUT_ONLY = "Predict the discharge functional outcome (good = mRS 0-2, poor = mRS 3-6)."


def clinical_text(r):
    sex = {"M": "man", "F": "woman"}.get(r.get("sex"), "patient")
    s = f"{int(r['age'])}-year-old {sex}" if pd.notna(r.get("age")) else f"Adult {sex}"
    s += f", admission NIHSS {int(r['nihss'])}" if pd.notna(r.get("nihss")) else ", admission NIHSS unknown"
    if pd.notna(r.get("priorstroke")):
        s += f", prior stroke {'yes' if r['priorstroke'] == 1 else 'no'}"
    return s + "."


def build_items(k):
    idx = pd.read_csv(WORK / "cache" / "index.csv")
    coh = pd.read_csv(RES / "cohort_all.csv")
    R = pd.read_csv(RES / "vlm" / "reports.csv")
    d = coh.merge(idx[["row", "participant_id"]], on="participant_id").merge(
        R[["participant_id", "report", "vlm_fold"]], on="participant_id", how="left")
    d = d[d.report.notna() | d.labelled].copy()
    d["fold_k"] = np.where(d.labelled, d.fold, d.vlm_fold).astype(int)
    items = []
    for _, r in d.iterrows():
        has_rep, lab = isinstance(r.report, str), bool(r.labelled)
        instr = INSTR_OUT if (has_rep and lab) else INSTR_REP if has_rep else INSTR_OUT_ONLY
        tgt = (r.report if has_rep else "")
        if lab:
            tgt = (tgt + "\n" if has_rep else "") + f"OUTCOME: {'poor' if r.y_poor == 1 else 'good'}"
        items.append(dict(pid=r.participant_id, row=int(r.row), test=int(r.fold_k) == k, labelled=lab, has_rep=has_rep,
                          y=r.y_poor, clin=clinical_text(r), instr=instr, target=tgt))
    return items


class VLM(nn.Module):
    def __init__(self, llm, tok, vis_dim=512):
        super().__init__()
        self.llm, self.tok = llm, tok
        H = llm.get_input_embeddings().weight.shape[1]
        self.proj = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, H), nn.GELU(), nn.Linear(H, H))
        self._cache = {}

    def prompt_ids(self, clin, instr):
        key = (clin, instr)
        if key not in self._cache:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Diffusion MRI (TRACE + ADC): <image>\nClinical: {clin}\n{instr}"}]
            try:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:                         # models without a system role
                msgs = [{"role": "user", "content": SYSTEM + "\n" + msgs[1]["content"]}]
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            pre, post = text.split("<image>")
            self._cache[key] = (self.tok(pre, add_special_tokens=False).input_ids, self.tok(post, add_special_tokens=False).input_ids)
        return self._cache[key]

    def embed(self, ids):
        E = self.llm.get_input_embeddings()
        return E(torch.tensor(ids, device=E.weight.device))

    def seq(self, vis, clin, instr, tail_ids=()):
        pre, post = self.prompt_ids(clin, instr)
        v = self.proj(vis.float()).to(self.llm.get_input_embeddings().weight.dtype)
        parts = [self.embed(pre), v, self.embed(post)] + ([self.embed(list(tail_ids))] if len(tail_ids) else [])
        return torch.cat(parts, 0), len(pre) + len(v) + len(post)


def pad_batch(seqs, labels=None, left=False):
    L = max(s.shape[0] for s in seqs); D = seqs[0].shape[1]; dev = seqs[0].device
    E = torch.zeros(len(seqs), L, D, dtype=seqs[0].dtype, device=dev)
    M = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    Y = torch.full((len(seqs), L), -100, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        n = s.shape[0]; sl = slice(L - n, L) if left else slice(0, n)
        E[i, sl] = s; M[i, sl] = 1
        if labels is not None:
            Y[i, sl] = labels[i]
    return E, M, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--proj-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--_test_model", default=None, help=argparse.SUPPRESS)   # internal offline test hook
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    (RES / "vlm").mkdir(parents=True, exist_ok=True)

    for k in (a.folds[:1] if a.quick else a.folds):
        t0 = time.time(); torch.manual_seed(k)
        if a._test_model:
            tok, llm = a._test_model
        else:
            tok = AutoTokenizer.from_pretrained(a.llm)
            llm = AutoModelForCausalLM.from_pretrained(a.llm, torch_dtype=dt)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        llm.to(dev)
        if dev.type == "cuda":
            llm.gradient_checkpointing_enable(); llm.enable_input_require_grads()
        llm = get_peft_model(llm, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        model = VLM(llm, tok).to(dev)
        feats = np.load(WORK / "vlm" / f"feats_fold{k}.npy", mmap_mode="r")
        items = build_items(k)
        tr = [it for it in items if not it["test"]]; te = [it for it in items if it["test"]]
        if a.quick:
            tr, te = tr[:32], te[:16]
        eos = tok.eos_token_id
        opt = torch.optim.AdamW([{"params": [p for p in llm.parameters() if p.requires_grad], "lr": a.lr},
                                 {"params": model.proj.parameters(), "lr": a.proj_lr}], weight_decay=0.0)
        steps = max(1, a.epochs * math.ceil(len(tr) / a.batch))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
        print(f"\n[vlm llm] fold {k}: {a.llm} | train {len(tr)} | test {len(te)} | {steps} steps", flush=True)
        rng = np.random.default_rng(k); step = 0
        for ep in range(a.epochs):
            model.train(); order = rng.permutation(len(tr)); tl = []
            for s in range(0, len(order), a.batch):
                seqs, labs = [], []
                for j in order[s:s + a.batch]:
                    it = tr[j]
                    tgt = tok(it["target"], add_special_tokens=False).input_ids + [eos]
                    vis = torch.from_numpy(np.asarray(feats[it["row"]], dtype=np.float32)).to(dev)
                    e, n_prompt = model.seq(vis, it["clin"], it["instr"], tgt)
                    seqs.append(e); labs.append(torch.tensor([-100] * n_prompt + tgt, device=dev))
                E, M, Y = pad_batch(seqs, labs)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    loss = llm(inputs_embeds=E, attention_mask=M, labels=Y).loss
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step(); sch.step(); tl.append(loss.item()); step += 1
                if step % 25 == 0:
                    print(f"  ep {ep} step {step}/{steps} loss {np.mean(tl[-25:]):.3f}", flush=True)
            print(f"  epoch {ep}: mean loss {np.mean(tl):.3f} | {(time.time()-t0)/60:.1f} min", flush=True)

        # ---------------- generation + outcome scoring on the held-out fold
        model.eval(); out = []
        id_poor = tok(" poor", add_special_tokens=False).input_ids[0]
        id_good = tok(" good", add_special_tokens=False).input_ids[0]
        with torch.no_grad():
            for s in range(0, len(te), a.gen_batch):
                chunk = te[s:s + a.gen_batch]
                vis = [torch.from_numpy(np.asarray(feats[it["row"]], dtype=np.float32)).to(dev) for it in chunk]
                gens = ["" for _ in chunk]
                need = [i for i, it in enumerate(chunk) if it["has_rep"]]
                if need:
                    E, M, _ = pad_batch([model.seq(vis[i], chunk[i]["clin"], chunk[i]["instr"])[0] for i in need], left=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        g = llm.generate(inputs_embeds=E, attention_mask=M, max_new_tokens=a.max_new, do_sample=False,
                                         pad_token_id=tok.pad_token_id, eos_token_id=eos)
                    for i, seq in zip(need, g):
                        gens[i] = tok.decode(seq, skip_special_tokens=True).split("OUTCOME:")[0].strip()
                p_poor = [np.nan] * len(chunk)
                lab_i = [i for i, it in enumerate(chunk) if it["labelled"]]
                if lab_i:
                    seqs = []
                    for i in lab_i:
                        tail = ((gens[i] + "\n") if chunk[i]["has_rep"] else "") + "OUTCOME:"
                        seqs.append(model.seq(vis[i], chunk[i]["clin"], chunk[i]["instr"],
                                              tok(tail, add_special_tokens=False).input_ids)[0])
                    E, M, _ = pad_batch(seqs, left=True)
                    pos = (M.cumsum(-1) - 1).clamp(min=0)          # correct positions under left padding
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        logits = llm(inputs_embeds=E, attention_mask=M, position_ids=pos).logits[:, -1].float()
                    pp = torch.softmax(logits[:, [id_good, id_poor]], 1)[:, 1].cpu().numpy()
                    for i, v in zip(lab_i, pp):
                        p_poor[i] = float(v)
                for it, gtxt, pv in zip(chunk, gens, p_poor):
                    out.append(dict(participant_id=it["pid"], labelled=it["labelled"], has_report=it["has_rep"],
                                    y_poor=it["y"], generated=gtxt, p_poor=pv))
                print(f"  generated {min(s + a.gen_batch, len(te))}/{len(te)}", flush=True)
        pd.DataFrame(out).to_csv(RES / "vlm" / f"gen_fold{k}{'_quick' if a.quick else ''}.csv", index=False)
        if not a.quick:
            llm.save_pretrained(str(WORK / "models" / f"vlm_lora_fold{k}"))
            torch.save(model.proj.state_dict(), WORK / "models" / f"vlm_proj_fold{k}.pt")
        print(f"[vlm llm] fold {k} done in {(time.time()-t0)/60:.1f} min", flush=True)
        del llm, model; torch.cuda.empty_cache() if dev.type == "cuda" else None


if __name__ == "__main__":
    main()
