import glob
import pandas as pd

from analisi_LLM import (load_results, add_metrics, tabella_modelli,
                         copertura_per_componente, bootstrap_ci)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)

PATTERN = "outputs/LLM/risultati_llm_20261027*.csv"
print("File letti:", glob.glob(PATTERN))

raw = load_results(PATTERN)
print("\nColonne:", list(raw.columns))
print("Righe:", len(raw))

print("\n=== 1. INTEGRITA' ===")
print(raw.groupby("modello").size())
print("immagini distinte:", raw["file"].nunique(), "| prompt distinti:", raw["prompt"].nunique())
print("risposte vuote:", (raw["risposta"].fillna("").str.len() == 0).sum())
chiave = ["label", "file"] if "label" in raw.columns else ["file"]
p = raw.pivot_table(index=chiave, columns="modello", values="risposta", aggfunc="first")
print("quota righe con risposta identica fra tutti i modelli:", round((p.nunique(axis=1) == 1).mean(), 3))

df = add_metrics(raw)

print("\n=== 2. TABELLA MODELLI ===")
tab = tabella_modelli(df)
print(tab)

print("\n=== 3. COPERTURA PER COMPONENTE (difetti) ===")
comp = copertura_per_componente(df)
print(comp)

print("\n=== 4. INTERVALLI DI CONFIDENZA (cop_difetti) ===")
print(bootstrap_ci(df))

print("\n=== 5. 15 RISPOSTE A CASO PER LA VALIDAZIONE A MANO ===")
for _, r in df.sample(15, random_state=0).iterrows():
    print(f"\n[{r.modello}] {r.file}")
    print("  difetti nel prompt:", r.issue_comps)
    print("  risposta:", r.risposta)
    print("  cop_difetti:", r.cop_difetti)

tab.to_csv("outputs/LLM/tabella_modelli.csv")
comp.to_csv("outputs/LLM/copertura_per_componente.csv")
print("\nSalvate: outputs/LLM/tabella_modelli.csv e copertura_per_componente.csv")