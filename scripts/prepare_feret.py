"""Prepara FERET per OFIQ: sceglie le immagini, le converte in PNG, le divide in blocchi
(shard) senza spezzare le identità e costruisce i metadati (manifest + xml ground truth).

Uso:
    python prepara_feret.py --out ~/prove/feret                       # tutto, solo pose fa/fb
    python prepara_feret.py --out ~/prove/feret --n-ids 100 --per-id 4  # campione
"""
import argparse
import bz2
import csv
import io
import os
import random
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

EXTS = (".ppm.bz2", ".ppm")          # accetta sia compressi che già decompressi


# ---------- 1) immagini ----------
def parse_name(path):
    stem = path.name
    for e in EXTS:
        if stem.endswith(e):
            stem = stem[: -len(e)]
            break
    parts = stem.split("_")
    return {
        "stem": stem,
        "id": parts[0],
        "date": parts[1] if len(parts) > 1 else "",
        "pose": parts[2] if len(parts) > 2 else "",
        "flag": parts[3] if len(parts) > 3 else "",
    }


def collect(root, dvds, poses):
    by_id = {}
    for dvd in dvds:
        img_root = Path(root) / dvd / "data" / "images"
        if not img_root.exists():
            print("manca:", img_root)
            continue
        for sub in sorted(os.listdir(img_root)):
            if not (img_root / sub).is_dir():          # salta indexFa.txt ecc.
                continue
            for f in sorted(os.listdir(img_root / sub)):
                if not f.endswith(EXTS):
                    continue
                info = parse_name(img_root / sub / f)
                if poses and info["pose"] not in poses:
                    continue
                info["src"] = str(img_root / sub / f)
                by_id.setdefault(info["id"], []).append(info)
    return by_id


def convert(job):
    src, dst = job
    if Path(dst).exists():
        return dst
    opener = bz2.open if str(src).endswith(".bz2") else open
    with opener(src, "rb") as f:
        im = Image.open(io.BytesIO(f.read()))
        im.convert("RGB").save(dst)
    return dst


# ---------- 2) xml ground truth ----------
def flatten(el, prefix=""):
    out = dict((f"{prefix}{k}", v) for k, v in el.attrib.items())
    kids = list(el)
    if not kids:
        if el.text and el.text.strip():
            out[prefix.rstrip("_") or el.tag] = el.text.strip()
        return out
    for c in kids:
        out.update(flatten(c, f"{prefix}{c.tag}_"))
    return out


def xml_to_rows(path):
    root = ET.parse(path).getroot()
    recs = list(root)
    while len(recs) == 1 and len(list(recs[0])) > 1:      # scende nei contenitori
        recs = list(recs[0])
    return [flatten(r) for r in recs]


def write_csv(rows, path):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return cols


def norm(v):
    v = str(v).strip()
    for e in (".ppm.bz2", ".ppm", ".bz2", ".tif", ".png"):
        if v.endswith(e):
            v = v[: -len(e)]
    return v.split("/")[-1]


def find_key(rows, stems):
    """Colonna dell'xml i cui valori coincidono di più con i nomi delle immagini."""
    best, best_n = None, 0
    for col in rows[0]:
        n = sum(1 for r in rows if norm(r.get(col, "")) in stems)
        if n > best_n:
            best, best_n = col, n
    return best, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/datasets/Feret")
    ap.add_argument("--dvds", nargs="+", default=["dvd1", "dvd2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", nargs="+", default=["fa", "fb"], help="'all' per tutte")
    ap.add_argument("--n-ids", type=int, default=0, help="0 = tutte le identità")
    ap.add_argument("--per-id", type=int, default=0, help="0 = tutte le immagini")
    ap.add_argument("--shard", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-convert", action="store_true", help="solo manifest e xml")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    poses = None if args.poses == ["all"] else set(args.poses)

    by_id = collect(args.root, args.dvds, poses)
    ids = sorted(by_id)
    print("identità trovate:", len(ids), "| immagini:", sum(len(v) for v in by_id.values()))
    if args.n_ids:
        ids = sorted(random.sample(ids, min(args.n_ids, len(ids))))

    # blocchi: un'identità non viene mai divisa tra due blocchi
    manifest, jobs, shard_i, in_shard = [], [], 0, 0
    for i in ids:
        imgs = by_id[i]
        if args.per_id:
            imgs = random.sample(imgs, min(args.per_id, len(imgs)))
        if in_shard and in_shard + len(imgs) > args.shard:
            shard_i, in_shard = shard_i + 1, 0
        d = out / "images" / f"shard_{shard_i:05d}"
        d.mkdir(parents=True, exist_ok=True)
        for info in imgs:
            dst = d / f"{info['stem']}.png"
            jobs.append((info["src"], str(dst)))
            manifest.append({"shard": d.name, "file": dst.name, "id": info["id"],
                             "date": info["date"], "pose": info["pose"],
                             "flag": info["flag"], "stem": info["stem"], "src": info["src"]})
        in_shard += len(imgs)
    print("immagini scelte:", len(manifest), "| blocchi:", shard_i + 1)

    write_csv(manifest, out / "manifest.csv")

    if not args.no_convert:
        with Pool(args.workers) as p:
            for k, _ in enumerate(p.imap_unordered(convert, jobs, chunksize=16), 1):
                if k % 2000 == 0:
                    print("convertite:", k, "/", len(jobs), flush=True)

    # ---------- xml ----------
    stems = {m["stem"] for m in manifest} | {m["stem"].rsplit("_", 1)[0] for m in manifest}
    for dvd in args.dvds:
        gt = Path(args.root) / dvd / "data" / "ground_truths" / "xml"
        for name in ("subjects", "recordings"):
            p = gt / f"{name}.xml"
            if not p.exists():
                continue
            rows = xml_to_rows(p)
            cols = write_csv(rows, out / f"{dvd}_{name}.csv")
            print(f"{dvd}/{name}.xml -> {len(rows)} record, colonne: {cols[:12]}")
            if name == "recordings" and rows:
                key, n = find_key(rows, stems)
                if key:
                    print(f"  chiave per unire alle immagini: '{key}' ({n} corrispondenze)")
                else:
                    print("  nessuna colonna coincide con i nomi delle immagini: guarda il csv")

    print("fatto. cartella:", out)


if __name__ == "__main__":
    main()