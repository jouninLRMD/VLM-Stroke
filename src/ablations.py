"""
Ablations, grounding tests and uncertainty for the grounded vision-language model.

Every variant uses the same folds, prompts, targets and training settings as vlm.py; only the named factor changes.
One model per fold is trained (or, with --eval-only, the adapters saved by vlm.py are reloaded).

  --tag full --eval-only  the main model, re-evaluated with extra inference tests:
                            --shuffle   visual tokens taken from another held-out scan (counterfactual grounding test)
                            --samples N N sampled reports per scan (temperature 0.7) for self-consistency uncertainty
                            outcome probability also read without writing the report first (p_poor_direct)
  --no-image              language prior only: the prompt has no visual tokens
  --no-clin               the clinical sentence is removed from the prompt
  --pooled-only           only the pooled visual token (1 instead of 18)
  --feats imnet           visual tokens from a frozen ImageNet ResNet-18 that never saw the findings (no grounding);
                          create them first with --dump-imnet
  --llm <hf id>           a different language model size (e.g. Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-3B-Instruct)
  --seed s                training seed (repeats of the full model)
  --scratch               no language pretraining: a small decoder (4 layers, width 256, same tokenizer) trained from
                          random initialization with all weights updated
Outputs: results/vlm/abl/gen_<tag>_fold{k}.csv ; adapters in <work>/models/vlm_<tag>_fold{k}
Examples (reproduce.py runs all of them):
  python src/ablations.py --dump-imnet
  python src/ablations.py --tag noimage --no-image
  python src/ablations.py --tag full --eval-only --shuffle
"""
import argparse, json, math, time
try:
    from PIL import Image  # noqa: F401  (on Windows, load Pillow before torch)
except Exception:
    pass
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import RES, WORK
from vlm import SYSTEM, INSTR_OUT_ONLY, build_items, VLM, pad_batch

ABL = RES / "vlm" / "abl"


def dump_imnet():
    """Frozen ImageNet ResNet-18 slice features (no training on the findings): 17 slices + mean-pooled token."""
    import torchvision
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    to3 = nn.Conv2d(2, 3, 1)
    with torch.no_grad():
        to3.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]]).view(3, 2, 1, 1)); to3.bias.zero_()
    m, to3 = m.to(dev).eval(), to3.to(dev)
    X = np.load(WORK / "cache" / "X.npy", mmap_mode="r"); N, nz = X.shape[0], X.shape[-1]
    z = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist())); S = len(z)
    out = np.lib.format.open_memmap(WORK / "vlm" / "feats_imnet.npy", mode="w+", dtype=np.float16, shape=(N, S + 1, 512))
    with torch.no_grad():
        for i in range(0, N, 8):
            x = torch.from_numpy(np.asarray(X[i:i + 8], dtype=np.float32)).to(dev)[:, :, :, :, z]
            B, C, D, H, _ = x.shape
            f = m(to3(x.permute(0, 4, 1, 2, 3).reshape(B * S, C, D, H))).view(B, S, -1)
            out[i:i + B] = torch.cat([f, f.mean(1, keepdim=True)], 1).float().cpu().numpy().astype(np.float16)
    out.flush(); print("ImageNet features:", out.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--proj-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--max-new", type=int, default=150)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--no-image", action="store_true"); ap.add_argument("--no-clin", action="store_true")
    ap.add_argument("--pooled-only", action="store_true")
    ap.add_argument("--feats", choices=["grounded", "imnet"], default="grounded")
    ap.add_argument("--eval-only", action="store_true", help="reload the adapters of vlm.py (tag full) or of an earlier run")
    ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dump-imnet", action="store_true")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--_test_model", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.dump_imnet:
        dump_imnet(); return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    amp = dev.type == "cuda"
    ABL.mkdir(parents=True, exist_ok=True); (WORK / "models").mkdir(parents=True, exist_ok=True)
    clin_off = "not provided."

    for k in (a.folds[:1] if a.quick else a.folds):
        t0 = time.time(); seed = k if a.seed is None else 1000 * a.seed + k
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        if a._test_model:
            tok, llm = a._test_model()
        elif a.scratch:
            from transformers import AutoConfig
            tok = AutoTokenizer.from_pretrained(a.llm)
            cfg = AutoConfig.from_pretrained(a.llm)
            cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 256, 1024, 4
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.tie_word_embeddings = 4, 2, True
            if hasattr(cfg, "layer_types"):
                cfg.layer_types = cfg.layer_types[:4]
            if hasattr(cfg, "max_window_layers"):
                cfg.max_window_layers = 4
            llm = AutoModelForCausalLM.from_config(cfg).to(dt)
        else:
            tok = AutoTokenizer.from_pretrained(a.llm)
            llm = AutoModelForCausalLM.from_pretrained(a.llm, dtype=dt)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        llm.to(dev)
        if a.eval_only:
            src = "vlm_lora" if a.tag == "full" else f"vlm_{a.tag}"
            llm = PeftModel.from_pretrained(llm, str(WORK / "models" / f"{src}_fold{k}")).to(dev)
        elif not a.scratch:
            if dev.type == "cuda":
                llm.gradient_checkpointing_enable(); llm.enable_input_require_grads()
            llm = get_peft_model(llm, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
        model = VLM(llm, tok).to(dev)
        if a.eval_only:
            src = "vlm_proj" if a.tag == "full" else f"vlm_{a.tag}_proj"
            model.proj.load_state_dict(torch.load(WORK / "models" / f"{src}_fold{k}.pt", map_location=dev))
        fpath = WORK / "vlm" / ("feats_imnet.npy" if a.feats == "imnet" else f"feats_fold{k}.npy")
        feats = np.load(fpath, mmap_mode="r")

        def vis_of(row):
            v = np.asarray(feats[row], dtype=np.float32)
            v = v[-1:] if a.pooled_only else v
            return torch.from_numpy(v).to(dev)

        def seq(it, row, instr=None, tail=()):
            clin = clin_off if a.no_clin else it["clin"]
            v = vis_of(row)[:0] if a.no_image else vis_of(row)
            return model.seq(v, clin, instr or it["instr"], tail)

        items = build_items(k)
        tr = [it for it in items if not it["test"]]; te = [it for it in items if it["test"]]
        if a.quick:
            tr, te = tr[:32], te[:16]
        eos = tok.eos_token_id

        if not a.eval_only:
            opt = torch.optim.AdamW([{"params": [p for p in llm.parameters() if p.requires_grad], "lr": a.lr},
                                     {"params": model.proj.parameters(), "lr": a.proj_lr}], weight_decay=0.0)
            steps = max(1, a.epochs * math.ceil(len(tr) / a.batch))
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))
            print(f"\n[vlm {a.tag}] fold {k}: {a.llm} | train {len(tr)} | test {len(te)} | {steps} steps", flush=True)
            step = 0
            for ep in range(a.epochs):
                model.train(); order = rng.permutation(len(tr)); tl = []
                for s in range(0, len(order), a.batch):
                    seqs, labs = [], []
                    for j in order[s:s + a.batch]:
                        it = tr[j]
                        tgt = tok(it["target"], add_special_tokens=False).input_ids + [eos]
                        e, n_prompt = seq(it, it["row"], tail=tgt)
                        seqs.append(e); labs.append(torch.tensor([-100] * n_prompt + tgt, device=dev))
                    E, M, Y = pad_batch(seqs, labs)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        loss = llm(inputs_embeds=E, attention_mask=M, labels=Y).loss
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                    opt.step(); sch.step(); tl.append(loss.item()); step += 1
                    if step % 50 == 0:
                        print(f"  ep {ep} step {step}/{steps} loss {np.mean(tl[-50:]):.3f}", flush=True)
                print(f"  epoch {ep}: mean loss {np.mean(tl):.3f} | {(time.time()-t0)/60:.1f} min", flush=True)
        else:
            print(f"\n[vlm {a.tag} eval-only] fold {k}: test {len(te)}", flush=True)

        # ---------------- inference
        model.eval(); out = []
        id_poor = tok(" poor", add_special_tokens=False).input_ids[0]
        id_good = tok(" good", add_special_tokens=False).input_ids[0]
        rep_idx = [i for i, it in enumerate(te) if it["has_rep"]]
        donor = {i: i for i in range(len(te))}
        if a.shuffle and len(rep_idx) > 1:                      # derangement among held-out scans with reports
            perm = np.array(rep_idx)
            while True:
                p2 = rng.permutation(perm)
                if not np.any(p2 == perm):
                    break
            donor.update(dict(zip(perm.tolist(), p2.tolist())))

        def gen(idxs, sample=False):
            E, M, _ = pad_batch([seq(te[i], te[donor[i]]["row"])[0] for i in idxs], left=True)
            kw = dict(do_sample=True, temperature=a.temperature, top_p=0.95) if sample else dict(do_sample=False)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                g = llm.generate(inputs_embeds=E, attention_mask=M, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id,
                                 eos_token_id=eos, **kw)
            return [tok.decode(x, skip_special_tokens=True).split("OUTCOME:")[0].strip() for x in g]

        def p_poor(idxs, texts, direct=False):
            seqs = []
            for i, txt in zip(idxs, texts):
                if direct:
                    seqs.append(seq(te[i], te[donor[i]]["row"], INSTR_OUT_ONLY, tok("OUTCOME:", add_special_tokens=False).input_ids)[0])
                else:
                    tail = ((txt + "\n") if te[i]["has_rep"] else "") + "OUTCOME:"
                    seqs.append(seq(te[i], te[donor[i]]["row"], tail=tok(tail, add_special_tokens=False).input_ids)[0])
            E, M, _ = pad_batch(seqs, left=True)
            pos = (M.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg = llm(inputs_embeds=E, attention_mask=M, position_ids=pos).logits[:, -1].float()
            return torch.softmax(lg[:, [id_good, id_poor]], 1)[:, 1].cpu().numpy()

        for s in range(0, len(te), a.gen_batch):
            idx = list(range(s, min(s + a.gen_batch, len(te))))
            gens = {i: "" for i in idx}; samples = {i: [] for i in idx}
            need = [i for i in idx if te[i]["has_rep"]]
            if need:
                for i, t in zip(need, gen(need)):
                    gens[i] = t
                for _ in range(a.samples):
                    for i, t in zip(need, gen(need, sample=True)):
                        samples[i].append(t)
            lab = [i for i in idx if te[i]["labelled"]]
            pp, pd_ = {}, {}
            if lab:
                pp = dict(zip(lab, p_poor(lab, [gens[i] for i in lab])))
                pd_ = dict(zip(lab, p_poor(lab, [""] * len(lab), direct=True)))
            for i in idx:
                it = te[i]
                out.append(dict(participant_id=it["pid"], labelled=it["labelled"], has_report=it["has_rep"], y_poor=it["y"],
                                donor=te[donor[i]]["pid"], generated=gens[i], p_poor=pp.get(i, np.nan),
                                p_poor_direct=pd_.get(i, np.nan), samples=json.dumps(samples[i]) if a.samples else ""))
            print(f"  generated {idx[-1] + 1}/{len(te)}", flush=True)
        name = a.tag + ("_shuffle" if a.shuffle else "") + ("_quick" if a.quick else "")
        pd.DataFrame(out).to_csv(ABL / f"gen_{name}_fold{k}.csv", index=False)
        if not a.eval_only and not a.quick:
            llm.save_pretrained(str(WORK / "models" / f"vlm_{a.tag}_fold{k}"))
            torch.save(model.proj.state_dict(), WORK / "models" / f"vlm_{a.tag}_proj_fold{k}.pt")
        print(f"[vlm {a.tag}] fold {k} done in {(time.time()-t0)/60:.1f} min", flush=True)
        del llm, model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
