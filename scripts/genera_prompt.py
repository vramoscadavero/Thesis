import argparse
import json
from pathlib import Path

import polars as pl

from genera_soglie import read_many

# fatto -> (frase difetto, frase aspetto positivo)
FACTS = {
    "eyes_closed": ("eyes closed", "eyes open"),
    "face_occluded": ("face partly covered (for example by glasses)", "face unobstructed"),
    "eyes_covered": ("eyes partly covered", "eyes clearly visible"),
    "head_covering": ("head covering present", "no head covering"),
    "not_neutral": ("non-neutral expression", "neutral expression"),
    "not_neutral_tono": ("non-neutral expression", "neutral expression"),
    "mouth_open": ("mouth open", "mouth closed"),
    "blurry": ("blurry image", "sharp image"),
    "bad_exposure": ("badly exposed image", "well exposed image"),
    "uneven_light": ("uneven lighting on the face", "even lighting on the face"),
    "bad_background": ("non-uniform background", "uniform background"),
}
# gravita' decrescente: decide quali fatti restano quando sono troppi
PRIORITY = ["eyes_closed", "face_occluded", "eyes_covered", "head_covering", "not_neutral", "mouth_open",
            "blurry", "bad_exposure", "uneven_light", "bad_background"]
# fatti che non dipendono dalla geometria del volto frontale: restano anche con la testa molto girata
# (dalla mappa di correlazione su FERET: espressione, bocca, copricapo e i fattori ambientali)
ROBUST = {"head_covering", "not_neutral", "not_neutral_tono", "mouth_open", "blurry", "bad_exposure",
          "uneven_light", "bad_background"}
POSE_DEFAULT = {"turned": 98, "strongly_turned": 65}   # usati solo se mancano nel JSON


def load_facts(soglie_path, include_status=("validated",), include_unconfirmed=False):
    meta = json.loads(Path(soglie_path).read_text())
    facts, pose, seen = [], dict(POSE_DEFAULT), set()
    for r in meta["facts"]:
        if r["fact"] in pose and r["status"] == "validated":
            pose[r["fact"]] = r["threshold"]
            continue
        if r["fact"] not in FACTS or r["status"] not in include_status:
            continue
        if r.get("needs_confirmation") and not include_unconfirmed:
            continue
        if r["component"] in seen or r["fact"] not in PRIORITY and r["fact"] != "not_neutral_tono":
            continue                                    # una sola regola per componente (la prima validata)
        seen.add(r["component"])
        facts.append({"fact": r["fact"], "component": r["component"], "threshold": float(r["threshold"])})
    return facts, pose


def pose_level(yaw, pose):
    if yaw is None or yaw != yaw:
        return "unknown"
    if yaw < pose["strongly_turned"]:
        return "strong"
    if yaw < pose["turned"]:
        return "mild"
    return "ok"


def build_facts(row, facts, pose, max_issues, max_good):
    level = pose_level(row.get("HeadPoseYawFrontal.scalar"), pose)
    issues, good = [], []                              # liste di nomi-fatto
    if level == "strong":
        issues_txt = ["head strongly turned (not facing the camera)"]
    elif level == "mild":
        issues_txt = ["head slightly turned"]
    else:
        issues_txt = []
    good_txt = ["head facing the camera"] if level == "ok" else []
    cand_i, cand_g = [], []
    for f in facts:
        v = row.get(f"{f['component']}.scalar")
        if v is None or v != v:
            continue
        if level == "strong" and f["fact"] not in ROBUST:
            continue
        (cand_i if v < f["threshold"] else cand_g).append(f["fact"])
    rank = lambda name: PRIORITY.index(name) if name in PRIORITY else len(PRIORITY)
    cand_i.sort(key=rank)
    cand_g.sort(key=rank)
    n_i = max(0, max_issues - len(issues_txt))
    n_g = max(0, max_good - len(good_txt))
    issues += cand_i[:n_i]
    good += cand_g[:n_g]
    return (issues_txt + [FACTS[x][0] for x in issues], good_txt + [FACTS[x][1] for x in good],
            ([f"pose_{level}"] if issues_txt else []) + issues, (["pose_ok"] if good_txt else []) + good, level)


def build_prompt(issues, good):
    return ("Describe a passport-style face photo in at most 2 sentences (max 45 words), using ONLY the facts "
            "below. Do not mention numbers, and do not infer identity, age, gender, emotion or any detail that "
            "is not listed. If Issues is none, say that no problems were found.\n"
            f"Issues: {'; '.join(issues) or 'none'}.\n"
            f"Good aspects: {'; '.join(good) or 'none'}.")


def genera_prompt_soglie(ofiq_paths, soglie_path, out_csv, max_issues=4, max_good=3, split="test",
                         include_status=("validated",), include_unconfirmed=False):
    facts, pose = load_facts(soglie_path, include_status, include_unconfirmed)
    df = read_many(ofiq_paths)
    scal = ["Filename"] + [c for c in df.columns if c.endswith(".scalar")]
    rows = []
    for row in df.select(scal).iter_rows(named=True):
        i_txt, g_txt, i_f, g_f, level = build_facts(row, facts, pose, max_issues, max_good)
        rows.append({"path": row["Filename"], "file": Path(row["Filename"]).name, "split": split,
                     "issues": "; ".join(i_txt) or "none", "good": "; ".join(g_txt) or "none",
                     "issues_list": json.dumps(i_txt), "good_list": json.dumps(g_txt),
                     "issue_facts": json.dumps(i_f), "good_facts": json.dumps(g_f), "pose_level": level,
                     "prompt": build_prompt(i_txt, g_txt)})
    out = pl.DataFrame(rows)
    Path(out_csv).expanduser().parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(Path(out_csv).expanduser())
    print(f"fatti usati: {[f['fact'] for f in facts]}")
    print(f"soglie posa: {pose}")
    print(f"prompt scritti: {out.height} -> {out_csv}")
    print("livelli di posa:", dict(out['pose_level'].value_counts().iter_rows()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ofiq", nargs="+", required=True, help="uno o piu' CSV OFIQ")
    ap.add_argument("--soglie", required=True, help="soglie_v*.json di genera_soglie.py (quello completo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-issues", type=int, default=4)
    ap.add_argument("--max-good", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--include-status", nargs="*", default=["validated"], help="es. validated weak")
    ap.add_argument("--include-unconfirmed", action="store_true")
    a = ap.parse_args()
    genera_prompt_soglie(a.ofiq, a.soglie, a.out, a.max_issues, a.max_good, a.split, tuple(a.include_status),
                         a.include_unconfirmed)


if __name__ == "__main__":
    main()