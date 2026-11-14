import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from genera_prompt import FRASI3

LIMIT_WORDS = 45
LIMIT_SENTENCES = 2

PHRASE2COMP = {}
for comp, phrases in FRASI3.items():
    for ph in phrases:
        if ph:
            PHRASE2COMP[ph] = comp

TOPICS = {
    "EyesOpen": [r"\beyes?\b", r"open-eyed", r"closed"],
    "ExpressionNeutrality": [r"expression", r"smil", r"neutral"],
    "NoHeadCoverings": [r"head covering", r"headwear", r"\bhat\b", r"scarf", r"hijab", r"covered", r"uncovered", r"bare head"],
    "BackgroundUniformity": [r"background"],
    "IlluminationUniformity": [r"lighting", r"illuminat", r"\blit\b", r"shadow"],
    "LuminanceMean": [r"exposed", r"exposure", r"\bdark", r"bright", r"luminan"],
    "HeadSize": [r"head size", r"size of the head", r"head (is |was )?(too |slightly |adequately |properly )?(small|large|big|sized)",
                 r"small head", r"head of adequate", r"head in the frame", r"head within the frame", r"sized head"],
    "MarginAboveTheFace": [r"\babove\b", r"headroom"],
    "MarginBelowTheFace": [r"\bbelow\b"],
}

EXTRA_TERMS = [r"unsettling", r"moody", r"dramatic", r"beautiful", r"handsome", r"\byoung\b", r"\bold\b",
               r"\bman\b", r"\bwoman\b", r"\bmale\b", r"\bfemale\b", r"glasses", r"smiling", r"serious"]


def load_results(pattern):
    dfs = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        d["source_file"] = Path(f).name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def parse_prompt(prompt):
    issues = re.search(r"Issues:\s*(.*)", prompt)
    good = re.search(r"Good aspects:\s*(.*)", prompt)

    def comps(m):
        if not m or m.group(1).strip() == "none":
            return []
        return [PHRASE2COMP[p.strip()] for p in m.group(1).split(";") if p.strip() in PHRASE2COMP]

    return comps(issues), comps(good)


def coverage(answer, components):
    if not components:
        return np.nan
    a = answer.lower()
    hits = sum(any(re.search(p, a) for p in TOPICS[c]) for c in components)
    return hits / len(components)


def per_component_hit(answer, components):
    a = answer.lower()
    return {c: any(re.search(p, a) for p in TOPICS[c]) for c in components}


def n_sentences(text):
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def add_metrics(df):
    df = df.copy()
    df["risposta"] = df["risposta"].fillna("").astype(str)
    parsed = df["prompt"].map(parse_prompt)
    df["issue_comps"] = parsed.map(lambda x: x[0])
    df["good_comps"] = parsed.map(lambda x: x[1])
    df["cop_difetti"] = [coverage(a, c) for a, c in zip(df.risposta, df.issue_comps)]
    df["cop_pregi"] = [coverage(a, c) for a, c in zip(df.risposta, df.good_comps)]
    df["parole"] = df.risposta.str.split().map(len)
    df["frasi"] = df.risposta.map(n_sentences)
    df["oltre_parole"] = df.parole > LIMIT_WORDS
    df["oltre_frasi"] = df.frasi > LIMIT_SENTENCES
    df["preambolo"] = df.risposta.str.contains(r"^\s*(?:here(?:'|’)?s|sure|certainly|okay|of course)", case=False, regex=True)
    df["numeri"] = df.risposta.str.contains(r"\d")
    df["extra"] = df.risposta.map(lambda a: sum(bool(re.search(p, a.lower())) for p in EXTRA_TERMS))
    df["token_clip_approx"] = df.risposta.map(lambda a: len(re.findall(r"\w+|[^\w\s]", a)))
    df["oltre_77"] = df.token_clip_approx > 75
    df["soggetto"] = df["file"].str.extract(r"^(S\d+)")[0].fillna(df["file"])
    return df


def tabella_modelli(df):
    cols = {
        "cop_difetti": ("cop_difetti", "mean"),
        "cop_pregi": ("cop_pregi", "mean"),
        "parole": ("parole", "mean"),
        "oltre_45_parole": ("oltre_parole", "mean"),
        "oltre_2_frasi": ("oltre_frasi", "mean"),
        "preambolo": ("preambolo", "mean"),
        "termini_extra": ("extra", "mean"),
        "oltre_77_tok": ("oltre_77", "mean"),
    }
    if "tempo_s" in df:
        cols["tempo_s"] = ("tempo_s", "mean")
    return df.groupby("modello").agg(**cols).round(3).sort_values("cop_difetti", ascending=False)


def copertura_per_componente(df):
    rows = []
    for _, r in df.iterrows():
        for c, hit in per_component_hit(r.risposta, r.issue_comps).items():
            rows.append({"modello": r.modello, "componente": c, "citato": hit})
    return pd.DataFrame(rows).pivot_table(index="componente", columns="modello", values="citato", aggfunc="mean").round(2)


def bootstrap_ci(df, col="cop_difetti", by="soggetto", n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m, g in df.groupby("modello"):
        per = g.groupby(by)[col].mean().dropna().to_numpy()
        means = [rng.choice(per, len(per)).mean() for _ in range(n_boot)]
        out[m] = (round(per.mean(), 3), round(np.percentile(means, 2.5), 3), round(np.percentile(means, 97.5), 3))
    return pd.DataFrame(out, index=["media", "ic95_basso", "ic95_alto"]).T