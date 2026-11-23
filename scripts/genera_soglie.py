import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import polars as pl

GRID = np.arange(1, 101)         
SAT_FRAC = 0.9           

STANDARD = [("HeadPoseYawFrontal", 5.0, "turned_icao"),
            ("HeadPosePitchFrontal", 5.0, "pitch_icao"),
            ("HeadPoseRollFrontal", 8.0, "roll_icao")]

PHRASES = {
    "eyes_covered": "eyes partly covered", "face_occluded": "face partly occluded",
    "turned": "head turned", "strongly_turned": "head strongly turned",
    "not_neutral": "non-neutral expression", "not_neutral_tono": "non-neutral expression",
    "mouth_open": "mouth open", "eyes_closed": "eyes closed", "head_covering": "head covering",
    "blurry": "blurry image", "uneven_light": "uneven lighting", "bad_exposure": "badly exposed image",
    "bad_background": "non-uniform background",
}


def read_ragged(path, sep):
    """Righe con il separatore dentro al Filename: le ultime N-1 colonne sono numeri, il resto e' il Filename."""
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split(sep)
        n = len(header)
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                rows.append(line.rsplit(sep, n - 1))
    bad = [r for r in rows if len(r) != n]
    if bad:
        raise ValueError(f"{path}: {len(bad)} righe con meno di {n} campi, file da controllare")
    df = pl.DataFrame(rows, schema=header, orient="row")
    num = [c for c in header if c != "Filename"]
    try:
        df = df.with_columns([pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).cast(pl.Float64).alias(c) for c in num])
    except Exception as e:
        raise ValueError(f"{path}: i campi in piu' non stanno nel Filename (valori non numerici): {e}")
    n_comma = df.filter(pl.col("Filename").str.contains(sep, literal=True)).height
    print(f"attenzione: {path}: {n_comma} Filename contengono '{sep}' (letti con il formato 'Filename + numeri')")
    return df


def read_ofiq(path):
    path = Path(path).expanduser()
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    try:
        df = pl.read_csv(path, separator=sep, infer_schema_length=None)
    except pl.exceptions.ComputeError:
        df = read_ragged(path, sep)
    df = df.select([c for c in df.columns if not c.startswith("Unnamed")])
    scal = [c for c in df.columns if c.endswith(".scalar")]
    stem = pl.col("Filename").str.split("/").list.last().str.replace(r"\.[A-Za-z0-9]+$", "")
    folder = pl.col("Filename").str.split("/").list.get(-2, null_on_oob=True)
    df = df.with_columns(stem.alias("stem"), folder.alias("folder"))
    return df.with_columns([pl.when(pl.col(c) >= 0).then(pl.col(c)).alias(c) for c in scal])  # -1 = errore


def read_many(paths):
    """Uno o piu' CSV OFIQ (uno per blocco) -> un solo DataFrame."""
    return pl.concat([read_ofiq(p) for p in paths], how="diagonal_relaxed")


def prep_glint(df):
    return df.with_columns(pl.lit("glint").alias("dataset"), pl.col("stem").alias("subject"))


def prep_tono(df):  
    return df.with_columns(pl.lit("tono").alias("dataset"), pl.col("stem").alias("subject"))


def prep_cfd(df, manifest):
    m = pl.read_csv(Path(manifest).expanduser(), infer_schema_length=None)
    m = m.select(pl.col("file").str.replace(r"\.[A-Za-z0-9]+$", "").alias("stem"),
                 pl.col("model").cast(pl.Utf8).alias("subject"), pl.col("expr").cast(pl.Utf8)).unique("stem")
    return df.join(m, on="stem", how="left").with_columns(pl.lit("cfd").alias("dataset"))


def prep_feret(df, recordings):
    rec = pl.concat([pl.read_csv(Path(p).expanduser(), infer_schema_length=None) for p in recordings],
                    how="diagonal_relaxed")
    rec = rec.select(
        pl.col("URL_relative").str.split("/").list.last().str.replace(r"\.ppm(\.bz2)?$", "").alias("stem"),
        pl.col("Subject_id").cast(pl.Utf8).alias("subject"),
        pl.col("Subject_Application_Face_Pose_name").alias("pose_gt"),
        pl.col("Subject_Application_Face_Pose_yaw").cast(pl.Float64).alias("yaw_nom"),
        pl.col("Subject_Application_Face_Wearing_glasses").alias("glasses")).unique("stem")
    return df.join(rec, on="stem", how="left").with_columns(pl.lit("feret").alias("dataset"))


def feret():
    return pl.col("dataset") == "feret"


def frontal():
    return feret() & pl.col("pose_gt").is_in(["fa", "fb"])


def cfd():
    return pl.col("dataset") == "cfd"


def tono(*folders):
    return (pl.col("dataset") == "tono") & pl.col("folder").is_in(list(folders))


def tono_other(*folders):  
    return (pl.col("dataset") == "tono") & ~pl.col("folder").is_in(list(folders))


GLASSES = dict(pos=lambda: frontal() & (pl.col("glasses") == "Yes"),
               neg=lambda: frontal() & (pl.col("glasses") == "No"))


RULES = [
    dict(comp="EyesVisible", fact="eyes_covered", **GLASSES),
    dict(comp="FaceOcclusionPrevention", fact="face_occluded", **GLASSES),
    dict(comp="HeadPoseYawFrontal", fact="turned",
         pos=lambda: feret() & ~pl.col("pose_gt").is_in(["fa", "fb"]), neg=lambda: frontal()),
    dict(comp="HeadPoseYawFrontal", fact="strongly_turned",
         pos=lambda: feret() & (pl.col("yaw_nom").abs() >= 60), neg=lambda: feret() & (pl.col("yaw_nom").abs() < 60)),
    dict(comp="ExpressionNeutrality", fact="not_neutral",
         pos=lambda: cfd() & pl.col("expr").is_in(["HO", "HC", "A", "F"]), neg=lambda: cfd() & (pl.col("expr") == "N")),
    dict(comp="MouthClosed", fact="mouth_open",
         pos=lambda: cfd() & (pl.col("expr") == "HO"), neg=lambda: cfd() & pl.col("expr").is_in(["N", "HC"])),
    dict(comp="ExpressionNeutrality", fact="not_neutral_tono", confirmed=False,
         pos=lambda: tono("sm"), neg=lambda: tono_other("sm")),
    dict(comp="EyesOpen", fact="eyes_closed", confirmed=False,
         pos=lambda: tono("ce", "ceg"), neg=lambda: tono_other("ce", "ceg")),
    dict(comp="NoHeadCoverings", fact="head_covering", confirmed=False,
         pos=lambda: tono("cap"), neg=lambda: tono_other("cap")),
    dict(comp="Sharpness", fact="blurry", confirmed=False,
         pos=lambda: tono("oof"), neg=lambda: tono_other("oof")),
    dict(comp="IlluminationUniformity", fact="uneven_light", confirmed=False,
         pos=lambda: tono("light"), neg=lambda: tono_other("light")),
    dict(comp="OverExposurePrevention", fact="bad_exposure", confirmed=False,
         pos=lambda: tono("expos"), neg=lambda: tono_other("expos")),
    dict(comp="BackgroundUniformity", fact="bad_background", confirmed=False,
         pos=lambda: tono("bkg"), neg=lambda: tono_other("bkg")),
]



def rates(pos, neg, grid=GRID):
    tpr = (pos[:, None] < grid[None, :]).mean(axis=0)
    fpr = (neg[:, None] < grid[None, :]).mean(axis=0)
    return tpr, fpr


def choose(pos, neg, min_tpr, max_fpr):
    tpr, fpr = rates(pos, neg)
    j = tpr - fpr
    ok = (tpr >= min_tpr) & (fpr <= max_fpr)
    score = np.where(ok, j, -np.inf) if ok.any() else j
    best = np.flatnonzero(score >= score.max() - 1e-9)     
    return int(GRID[best[len(best) // 2]])                  


def evaluate(x, y, subj, min_tpr, max_fpr, k, min_n, seed):
    rng = np.random.default_rng(seed)
    subjects = np.unique(subj)
    ts, tprs, fprs = [], [], []
    for _ in range(k):
        half = rng.permutation(subjects)[: len(subjects) // 2]
        a = np.isin(subj, half)
        xa, ya, xb, yb = x[a], y[a], x[~a], y[~a]
        if min((ya == 1).sum(), (ya == 0).sum(), (yb == 1).sum(), (yb == 0).sum()) < min_n:
            continue
        t = choose(xa[ya == 1], xa[ya == 0], min_tpr, max_fpr)
        tpr, fpr = rates(xb[yb == 1], xb[yb == 0], np.array([t]))
        ts.append(t)
        tprs.append(tpr[0])
        fprs.append(fpr[0])
    if not ts:
        return None
    t_final = int(round(float(np.median(ts))))
    tpr_all, fpr_all = rates(x[y == 1], x[y == 0], np.array([t_final]))
    return dict(threshold=t_final, splits_used=len(ts), threshold_sd=float(np.std(ts)),
                tpr_heldout=float(np.mean(tprs)), tpr_heldout_sd=float(np.std(tprs)),
                fpr_heldout=float(np.mean(fprs)), fpr_heldout_sd=float(np.std(fprs)),
                tpr_all=float(tpr_all[0]), fpr_all=float(fpr_all[0]))


def scalar_at_native(data, comp, deg, tol=1.0):
    nat, sca = f"{comp}.native", f"{comp}.scalar"
    if nat not in data.columns:
        return None
    s = data.filter((pl.col(nat).abs() - deg).abs() <= tol)[sca].drop_nulls()
    return float(s.median()) if s.len() >= 20 else None



def labelled_rules(data, args):
    rows = []
    for rule in RULES:
        comp, fact = rule["comp"], rule["fact"]
        col = f"{comp}.scalar"
        base = dict(component=comp, fact=fact, method="labels", needs_confirmation=not rule.get("confirmed", True))
        try:
            pos = data.filter(rule["pos"]()).select("subject", col).drop_nulls().with_columns(pl.lit(1).alias("y"))
            neg = data.filter(rule["neg"]()).select("subject", col).drop_nulls().with_columns(pl.lit(0).alias("y"))
        except Exception:
            rows.append({**base, "status": "no_data", "n_pos": 0, "n_neg": 0, "source": "dataset o colonna mancante"})
            continue
        base.update(n_pos=pos.height, n_neg=neg.height)
        if pos.height == 0 or neg.height == 0:
            rows.append({**base, "status": "no_data", "source": "nessun esempio"})
            continue
        sub = pl.concat([pos, neg])
        res = evaluate(sub[col].to_numpy().astype(float), sub["y"].to_numpy(), sub["subject"].to_numpy(),
                       args.min_tpr, args.max_fpr, args.k, args.min_n, args.seed)
        if res is None:
            rows.append({**base, "status": "insufficient_data", "source": f"meno di {args.min_n} esempi per classe/meta'"})
            continue
        ok = res["tpr_heldout"] >= args.min_tpr and res["fpr_heldout"] <= args.max_fpr
        dsets = "+".join(data.filter(rule["pos"]())["dataset"].unique().to_list())
        rows.append({**base, **res, "status": "validated" if ok else "weak",
                     "source": f"etichette {dsets} ({fact}), scelta su meta' dei soggetti, misura sull'altra (x{res['splits_used']})"})
    return rows


def standard_rows(data):
    rows = []
    for comp, deg, fact in STANDARD:
        s = scalar_at_native(data, comp, deg)
        if s is not None:
            rows.append(dict(component=comp, fact=fact, method="standard", threshold=int(round(s)),
                             status="standard", needs_confirmation=False, n_pos=None, n_neg=None,
                             source=f"ICAO +-{deg:g} gradi convertito in scalar con la mappatura OFIQ"))
    return rows


def fallback_rows(data, covered, ref, pct):
    d = data.filter(pl.col("dataset") == ref) if ref in data["dataset"].unique().to_list() else data
    rows = []
    for col in [c for c in data.columns if c.endswith(".scalar")]:
        comp = col[:-7]
        if comp in covered or comp.startswith("HeadPose"):
            continue
        x = d[col].drop_nulls().to_numpy().astype(float)
        if len(x) == 0:
            continue
        sat = float((x >= 99).mean())
        if sat >= SAT_FRAC:
            rows.append(dict(component=comp, fact=f"{comp}_low", method="binary", threshold=100,
                             status="saturated_binary", needs_confirmation=False, n_pos=None, n_neg=len(x),
                             source=f"saturo ({sat:.0%} a 100): scatta con qualsiasi calo"))
        else:
            rows.append(dict(component=comp, fact=f"{comp}_low", method="percentile",
                             threshold=round(float(np.percentile(x, pct)), 1), status="relative",
                             needs_confirmation=False, n_pos=None, n_neg=len(x),
                             source=f"percentile {pct:g} di {ref if d is not data else 'tutti i dataset'} (relativa)"))
    return rows


def compat_json(rows, include, unconfirmed=False):
    ok = {"validated", "standard"} | set(include)
    out = {}
    for r in rows:
        if r["component"].startswith("HeadPose") or r["status"] not in ok or r["component"] in out:
            continue
        if r.get("needs_confirmation") and not unconfirmed:  
            continue
        out[r["component"]] = {"method": "binary", "thresholds": [float(r["threshold"])],
                               "names": ["very_low", "high"], "source": r["source"], "status": r["status"],
                               "n": (r.get("n_pos") or 0) + (r.get("n_neg") or 0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glint", nargs="+", help="uno o piu' CSV OFIQ")
    ap.add_argument("--feret", nargs="+", help="uno o piu' CSV OFIQ")
    ap.add_argument("--feret-recordings", nargs="+")
    ap.add_argument("--cfd", nargs="+", help="uno o piu' CSV OFIQ")
    ap.add_argument("--cfd-manifest")
    ap.add_argument("--tono", nargs="+", help="uno o piu' CSV OFIQ")
    ap.add_argument("--out", required=True, help="es. soglie/soglie_v1.json")
    ap.add_argument("--min-tpr", type=float, default=0.7)
    ap.add_argument("--max-fpr", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=20, help="divisioni casuali per soggetto")
    ap.add_argument("--min-n", type=int, default=20, help="esempi minimi per classe in ogni meta'")
    ap.add_argument("--pct", type=float, default=10, help="percentile per i componenti senza etichette")
    ap.add_argument("--ref", default="glint", help="dataset di riferimento per i percentili")
    ap.add_argument("--include", nargs="*", default=[], help="stati extra nel file compatibile: weak relative saturated_binary")
    ap.add_argument("--include-unconfirmed", action="store_true", help="nel file compatibile includi anche le regole TONO non confermate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = []
    if args.glint:
        frames.append(prep_glint(read_many(args.glint)))
    if args.feret:
        frames.append(prep_feret(read_many(args.feret), args.feret_recordings))
    if args.cfd:
        frames.append(prep_cfd(read_many(args.cfd), args.cfd_manifest))
    if args.tono:
        frames.append(prep_tono(read_many(args.tono)))
    data = pl.concat(frames, how="diagonal_relaxed")
    print("immagini per dataset:", dict(data["dataset"].value_counts().iter_rows()))

    rows = labelled_rules(data, args) + standard_rows(data)
    covered = {r["component"] for r in rows if r["status"] in ("validated", "standard")}
    rows += fallback_rows(data, covered, args.ref, args.pct)
    for r in rows:
        r["issue_phrase"] = PHRASES.get(r["fact"])

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(version=out.stem, created=datetime.datetime.now().isoformat(timespec="seconds"),
                convention="scalar 0-100, alto = meglio; il fatto scatta se scalar < threshold",
                params=dict(min_tpr=args.min_tpr, max_fpr=args.max_fpr, k=args.k, min_n=args.min_n,
                            pct=args.pct, ref=args.ref, seed=args.seed))
    out.write_text(json.dumps({"meta": meta, "facts": rows}, indent=2, default=float))
    compat_path = out.with_name(out.stem + "_compat.json")
    compat_path.write_text(json.dumps(compat_json(rows, args.include, args.include_unconfirmed), indent=2))

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_fmt_str_lengths(40)
    show = pl.DataFrame(rows, infer_schema_length=None)
    cols = [c for c in ["component", "fact", "threshold", "status", "n_pos", "n_neg", "tpr_heldout", "fpr_heldout",
                        "needs_confirmation"] if c in show.columns]
    print(show.select(cols))
    print("scritti:", out, "e", compat_path)


if __name__ == "__main__":
    main()