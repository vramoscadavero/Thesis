"""Prepara il Chicago Face Database (CFD) per OFIQ: blocchi (shard) senza spezzare le persone,
manifest con i metadati ricavati dal nome e unione con il file Excel dei norming data.

Uso:
    python prepara_cfd.py --root /datasets/CFD --out ~/prove/cfd
    python prepara_cfd.py --root /datasets/CFD --out ~/prove/cfd_prova --n-ids 20
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png"}
EXPR = {"N": "neutral", "HO": "happy_open", "HC": "happy_closed", "A": "angry", "F": "fearful"}


def parse_name(path):
    """CFD-WF-001-003-N.jpg -> gruppo WF, modello WF-001, immagine 003, espressione N."""
    stem = path.stem
    p = stem.split("-")
    if len(p) >= 5 and p[0].upper() == "CFD":
        return {"stem": stem, "group": p[1], "model": f"{p[1]}-{p[2]}", "img_no": p[3],
                "expr": p[4], "expr_name": EXPR.get(p[4].upper(), p[4])}
    return {"stem": stem, "group": "", "model": stem, "img_no": "", "expr": "", "expr_name": ""}


def collect(images_root, expressions):
    by_model = {}
    for p in sorted(Path(images_root).rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        info = parse_name(p)
        if expressions and info["expr"].upper() not in expressions:
            continue
        info["src"] = str(p)
        by_model.setdefault(info["model"], []).append(info)
    return by_model


# ---------- norming data (xlsx) ----------
def read_sheet(xlsx, sheet):
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    head = 0
    for i in range(min(len(raw), 15)):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if any(c in ("model", "target", "id") for c in cells):
            head = i
            break
    df = raw.iloc[head + 1:].copy()
    cols, seen = [], {}
    for c in raw.iloc[head].tolist():                 # nomi colonna unici (l'Excel ne ripete)
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def norm_id(v):
    v = str(v).strip().upper()
    return v[4:] if v.startswith("CFD-") else v


def best_key(df, models):
    best, best_n = None, 0
    for c in df.columns:
        n = df[c].map(norm_id).isin(models).sum()
        if n > best_n:
            best, best_n = c, n
    return best, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/datasets/CFD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expr", nargs="+", default=["all"],
                    help="espressioni da tenere: N HO HC A F (default tutte)")
    ap.add_argument("--n-ids", type=int, default=0, help="0 = tutte le persone")
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--copy", action="store_true", help="copia i file invece di creare link")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root, out = Path(args.root), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    expressions = None if args.expr == ["all"] else {e.upper() for e in args.expr}

    by_model = collect(root / "Images", expressions)
    models = sorted(by_model)
    print("persone (modelli) trovate:", len(models), "| immagini:", sum(len(v) for v in by_model.values()))
    if args.n_ids:
        models = sorted(random.sample(models, min(args.n_ids, len(models))))

    manifest, shard_i, in_shard = [], 0, 0
    for m in models:
        imgs = by_model[m]
        if in_shard and in_shard + len(imgs) > args.shard:
            shard_i, in_shard = shard_i + 1, 0
        d = out / "images" / f"shard_{shard_i:05d}"
        d.mkdir(parents=True, exist_ok=True)
        for info in imgs:
            src = Path(info["src"])
            dst = d / (info["stem"] + src.suffix.lower())
            if not os.path.lexists(dst):
                if args.copy:
                    dst.write_bytes(src.read_bytes())
                else:
                    dst.symlink_to(src)
            manifest.append({"shard": d.name, "file": dst.name, "model": m,
                             "group": info["group"], "img_no": info["img_no"],
                             "expr": info["expr"], "expr_name": info["expr_name"], "src": info["src"]})
        in_shard += len(imgs)
    print("immagini scelte:", len(manifest), "| blocchi:", shard_i + 1)

    man = pd.DataFrame(manifest)
    man.to_csv(out / "manifest.csv", index=False)
    print("espressioni:", man["expr_name"].value_counts().to_dict())
    print("gruppi:", man["group"].value_counts().to_dict())

    # ---------- norming data ----------
    xlsx = next(iter(sorted(root.glob("*Norming*.xlsx"))), None)
    if xlsx is None:
        print("nessun file Excel dei norming data trovato in", root)
        return
    try:
        sheets = pd.ExcelFile(xlsx).sheet_names
    except ImportError:
        print("manca openpyxl: uv add openpyxl")
        return
    print("fogli Excel:", sheets)
    ids = set(man["model"])
    frames = []
    for s in sheets:
        df = read_sheet(xlsx, s)
        df.to_csv(out / f"norming_{s.replace(' ', '_')}.csv", index=False)
        if df.empty:
            continue
        key, n = best_key(df, ids)
        print(f"  foglio '{s}': {len(df)} righe, {df.shape[1]} colonne; chiave '{key}' ({n} corrispondenze)")
        if key and n > 0:
            frames.append(df.assign(model=df[key].map(norm_id), norming_sheet=s)
                            .drop(columns=[key]))
    if frames:
        norming = pd.concat(frames, ignore_index=True).drop_duplicates("model")
        meta = man.merge(norming, on="model", how="left")
        meta.to_csv(out / "metadata.csv", index=False)
        senza = meta.loc[meta["norming_sheet"].isna(), "model"].unique()
        print(f"metadata.csv: {meta.shape[1]} colonne, fogli usati: {len(frames)}")
        if len(senza):
            print("modelli senza dati di norming:", list(senza))
    else:
        print("nessun foglio si unisce al manifest: apri i csv norming_*.csv")
    print("fatto. cartella:", out)


if __name__ == "__main__":
    main()