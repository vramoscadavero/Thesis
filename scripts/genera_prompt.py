import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PERCENTILES = [5,10,25, 50, 75]
TEST_SIZE = 0.2
SEED = 42
CLASS_NAMES = ["worse","bad","very_low", "low", "medium", "high"]

FRASI3 = {
    "EyesOpen": ("eyes closed or mostly closed", "eyes partly closed", "eyes open"),
    "ExpressionNeutrality": ("clearly non-neutral expression", "slightly non-neutral expression", "neutral expression"),
    "NoHeadCoverings": ("head covering clearly visible", "partial head covering", "no head covering"),
    "BackgroundUniformity": ("cluttered, non-uniform background", "slightly uneven background", "uniform background"),
    "IlluminationUniformity": ("strongly uneven lighting on the face", "slightly uneven lighting", "even lighting on the face"),
    "LuminanceMean": ("image clearly too dark or too bright", "image slightly too dark or bright", "well exposed image"),
    "HeadSize": ("head too small in the frame", "head slightly small in the frame", "head of adequate size"),
    "MarginAboveTheFace": ("improper space above the head", "slightly improper space above the head", "proper space above the head"),
    "MarginBelowTheFace": ("improper space below the face", "slightly improper space below the face", "proper space below the face"),
}


def load_data(file_path):
    return pd.read_csv(file_path, sep=";").dropna(axis=1, how="all")


def select_data(df):
    scores = df.set_index("Filename").filter(like=".native")
    return scores.rename(columns=lambda c: c.replace(".native", ""))


def split_data(df, test_size, seed):
    return train_test_split(df, test_size=test_size, random_state=seed)


def compute_threshold(train, percentiles):
    res = {}
    for c in train.columns:
        y = train[c][train[c] >= 0].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        values = np.percentile(y, percentiles)
        res[c] = {
            "percentiles": {str(p): round(float(v), 2) for p, v in zip(percentiles, values)},
            "thresholds": values.tolist(),
            "n": int(len(y)),
        }
    return res


def save_threshold(res, upload_path, name=None):
    folder = Path(upload_path)
    folder.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = f"percentiles_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path = folder / name
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    return path


def load_threshold(path):
    with open(path) as f:
        return json.load(f)


def classify(component, score, thresholds, names=CLASS_NAMES):
    cuts = thresholds[component]["thresholds"]
    assert len(names) == len(cuts) + 1, "servono len(cuts)+1 nomi di fascia"
    return names[int(np.digitize(score, cuts))]


def build_facts(row, thresholds, phrases):
    neg, pos = [], []
    for c, (f_vl, f_low, f_ok) in phrases.items():
        if c not in thresholds or row[c] < 0:
            continue
        cl = classify(c, row[c], thresholds)
        if cl == "very_low":
            neg.append(f_vl)
        elif cl == "low":
            neg.append(f_low)
        else:
            pos.append(f_ok)
    return "; ".join(neg) or "none", "; ".join(pos) or "none"


def build_prompt(issues, good):
    return (
        "Describe a passport-style face photo in at most 2 sentences (max 45 words), "
        "using ONLY the facts below. Do not mention numbers or add other details.\n"
        f"Issues: {issues}\n"
        f"Good aspects: {good}"
    )


def build_dataset(scores, train_index, thresholds, phrases):
    rows = []
    for path, row in scores.iterrows():
        issues, good = build_facts(row, thresholds, phrases)
        rows.append({
            "file": Path(path).name,
            "split": "train" if path in train_index else "test",
            "issues": issues,
            "good": good,
            "prompt": build_prompt(issues, good),
        })
    return pd.DataFrame(rows)


def genera_prompt(csv_path, out_dir, percentiles=PERCENTILES,
                  test_size=TEST_SIZE, seed=SEED, phrases=FRASI3):
    df = load_data(csv_path)
    scores = select_data(df)
    train, _ = split_data(scores, test_size, seed)

    thresholds = compute_threshold(train, percentiles)
    save_threshold(thresholds, out_dir, name="percentiles.json")

    dataset = build_dataset(scores, set(train.index), thresholds, phrases)
    dataset.to_csv(Path(out_dir) / "prompt_ofiq_percentili.csv", index=False)
    return dataset