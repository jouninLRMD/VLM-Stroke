"""
Zero-shot general-purpose vision-language model as an external reference.

An off-the-shelf model (default Qwen/Qwen2.5-VL-3B-Instruct; any image-text-to-text model with a chat template can be
passed with --model) receives two montages of the same 17 axial levels used by the trained models (DWI trace and
relative ADC, neurological convention), the clinical sentence and an exact description of the report template, and
is asked to write the report and the outcome. No training.

Outputs: results/vlm/abl/gen_zeroshot_<name>_fold{k}.csv (same columns as vlm.py), one file per fold so that the same
evaluation code applies; montage examples in results/vlm/abl/montage_examples/.
Usage: python src/zeroshot.py [--model Qwen/Qwen2.5-VL-3B-Instruct] [--limit 20]
"""
import argparse, time
try:
    from PIL import Image
except Exception:
    Image = None
import numpy as np, pandas as pd, torch
from common import RES, WORK

ABL = RES / "vlm" / "abl"
TEMPLATE = (
    "Write the findings in exactly this format, on one line, choosing only from the options in brackets:\n"
    "FINDINGS: [Acute infarct | Multiple acute infarcts] on [the left side | the right side | both sides], involving the "
    "<territories> [territory | territories]. Acute lesion volume <number> mL ([small | medium | large | very large]). "
    "[Marked | Mild | No clear] diffusion restriction (mean relative ADC <number>). [Chronic infarct also present. | No chronic infarct.]\n"
    "Territory names to use: anterior cerebral artery (ACA); middle cerebral artery (MCA) cortical; deep "
    "lenticulostriate/anterior choroidal; posterior cerebral artery (PCA); thalamic (thalamoperforator); brainstem "
    "(basilar perforator); cerebellar. Join two territories with 'and'. For MCA cortical involvement add "
    "'(MCA branches: frontal, parietal, temporal, occipital, insular)' listing only the involved branches.\n"
    "Volume classes: small < 2 mL, medium 2 to 20 mL, large 20 to 70 mL, very large >= 70 mL. Diffusion restriction: "
    "marked if mean relative ADC < 0.8, mild if 0.8 to 1.0, no clear if >= 1.0.\n"
    "Then write a second line: OUTCOME: good (discharge mRS 0-2) or OUTCOME: poor (mRS 3-6).")
EXAMPLE = ("Example of the format: FINDINGS: Acute infarct on the right side, involving the middle cerebral artery (MCA) "
           "cortical territory (MCA branches: frontal, insular). Acute lesion volume 12.3 mL (medium). Mild diffusion "
           "restriction (mean relative ADC 0.88). No chronic infarct.\nOUTCOME: good")


def montage(vol, z, vmin, vmax, ncol=6, scale=2):
    tiles = []
    for k in z:
        s = np.rot90(np.asarray(vol[:, :, k], dtype=np.float32))           # RAS array -> anterior up, patient left on image left
        s = np.clip((s - vmin) / (vmax - vmin), 0, 1)
        tiles.append(s)
    h, w = tiles[0].shape; nrow = int(np.ceil(len(tiles) / ncol))
    M = np.zeros((nrow * h, ncol * w), np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol); M[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    im = Image.fromarray((M * 255).astype(np.uint8)).convert("RGB")
    return im.resize((im.width * scale, im.height * scale), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scans per fold (test run)")
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoVLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    tok = proc.tokenizer
    name = a.model.split("/")[-1].lower().replace(".", "")
    ABL.mkdir(parents=True, exist_ok=True); (ABL / "montage_examples").mkdir(exist_ok=True)

    idx = pd.read_csv(WORK / "cache" / "index.csv")
    coh = pd.read_csv(RES / "cohort_all.csv")
    R = pd.read_csv(RES / "vlm" / "reports.csv")
    d = coh.merge(idx[["row", "participant_id"]], on="participant_id").merge(R[["participant_id", "vlm_fold", "clinical_text"]],
                                                                          on="participant_id", how="left")
    d = d[d.vlm_fold.notna() | d.labelled].copy()
    d["fold_k"] = np.where(d.labelled, d.fold, d.vlm_fold).astype(int)
    d["has_rep"] = d.vlm_fold.notna()
    X = np.load(WORK / "cache" / "X.npy", mmap_mode="r"); nz = X.shape[-1]
    z = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist()))
    cand_poor = [tok(" poor", add_special_tokens=False).input_ids[0], tok("poor", add_special_tokens=False).input_ids[0]]
    cand_good = [tok(" good", add_special_tokens=False).input_ids[0], tok("good", add_special_tokens=False).input_ids[0]]

    def clin_text(r):
        if isinstance(r.clinical_text, str):
            return r.clinical_text
        sex = {"M": "man", "F": "woman"}.get(r.sex, "patient")
        return (f"{int(r.age)}-year-old {sex}" if pd.notna(r.age) else f"Adult {sex}") + \
               (f", admission NIHSS {int(r.nihss)}" if pd.notna(r.nihss) else "") + "."

    for k in range(5):
        t0 = time.time(); rows = []
        dk = d[d.fold_k == k]
        if a.limit:
            dk = dk.head(a.limit)
        for n, r in enumerate(dk.itertuples()):
            x = X[int(r.row)]
            imgs = [montage(x[0], z, -2, 4), montage(x[1], z, 0, 2)]
            if n < 3 and k == 0:
                for j, im in enumerate(imgs):
                    im.save(ABL / "montage_examples" / f"{r.participant_id}_{['trace', 'radc'][j]}.png")
            text = ("You see two montages of the same 17 axial diffusion MRI slices of a patient with acute ischemic stroke, "
                    "ordered from inferior (top left) to superior (bottom right), in neurological convention (the left side of "
                    "each image is the patient's left). Image 1: DWI trace (b = 1000), where acute infarcts are bright. Image 2: "
                    "relative ADC map, where restricted diffusion is dark (normal brain about 1.0).\n"
                    f"Clinical: {clin_text(r)}\n" + TEMPLATE + "\n" + EXAMPLE)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]}]
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inp = proc(text=[prompt], images=imgs, return_tensors="pt").to(dev)
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
            out = tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rep = out.split("OUTCOME:")[0].strip()
            rep = rep[rep.find("FINDINGS:"):] if "FINDINGS:" in rep else rep
            p = np.nan
            if r.labelled:
                inp2 = proc(text=[prompt + rep + "\nOUTCOME:"], images=imgs, return_tensors="pt").to(dev)
                with torch.no_grad():
                    lg = model(**inp2).logits[0, -1].float()
                pr = torch.logsumexp(lg[cand_poor], 0); gd = torch.logsumexp(lg[cand_good], 0)
                p = float(torch.sigmoid(pr - gd))
            rows.append(dict(participant_id=r.participant_id, labelled=bool(r.labelled), has_report=bool(r.has_rep),
                             y_poor=r.y_poor, generated=rep if r.has_rep else "", p_poor=p, raw=out))
            if n % 50 == 0:
                print(f"  fold {k}: {n + 1}/{len(dk)} | {(time.time()-t0)/60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(ABL / f"gen_zeroshot_{name}{'_test' if a.limit else ''}_fold{k}.csv", index=False)
        print(f"[zero-shot {name}] fold {k} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
