import subprocess
import time
import requests
import csv
import sys
from pathlib import Path
from datetime import datetime
from llama_server import avvia_server, wait_server, ferma_server
from genera_prompt import genera_prompt

modelli = [
    {"label": "LFM2-1.2B",   "repo": "LiquidAI/LFM2-1.2B-GGUF",                    "quant": "Q4_K_M"},
    {"label": "Gemma-3-1B",  "repo": "ggml-org/gemma-3-1b-it-GGUF",                "quant": "Q4_K_M"},
    {"label": "Qwen3-4B",    "repo": "Qwen/Qwen3-4B-GGUF",                         "quant": "Q4_K_M"},
    {"label": "Phi-4-mini",  "repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF", "quant": "Q4_K_M"},
    {"label": "Gemma-3-4B",  "repo": "ggml-org/gemma-3-4b-it-GGUF",                "quant": "Q4_K_M"},
    {"label": "Qwen3-8B",    "repo": "Qwen/Qwen3-8B-GGUF",                         "quant": "Q4_K_M"},
    {"label": "Llama-3.1-8B", "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "quant": "Q4_K_M"},
]

# provare con un batch di immagini circa sulle 200-300

SERVER_URL = "http://127.0.0.1:18080"

results_dir = Path("./outputs/LLM/glint")
results_dir.mkdir(parents=True, exist_ok=True)

def chiama_modello_testo(prompt):
    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": 150,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    inizio = time.time()
    risposta = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=720)
    tempo = time.time() - inizio
    risposta.raise_for_status()
    dati = risposta.json()
    scelta = dati["choices"][0]
    testo = scelta["message"]["content"].strip()
    print("   finish_reason:", scelta.get("finish_reason"), "| tokens:", dati["usage"]["completion_tokens"], "| inizio:", repr(testo[:80]))
    return testo, tempo, dati["usage"]["prompt_tokens"], dati["usage"]["completion_tokens"]

def main():
    dataset = genera_prompt("data/glint/ofiq_glint_10000.csv", results_dir)

    csv_path = results_dir / f"risultati_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    campi = ["timestamp", "modello", "file", "prompt", "risposta",
             "num_parole", "tempo_s", "prompt_tokens", "completion_tokens"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campi)
        writer.writeheader()

        for modello in modelli:
            print(f"\n\n=== {modello['label']} ===")
            processo = avvia_server(modello["repo"], modello["quant"], results_dir)
            try:
                if not wait_server(3600, SERVER_URL, processo):
                    print("Server non pronto, salto questo modello. Vedi il log in", results_dir)
                    continue

                print(requests.get(f"{SERVER_URL}/v1/models", timeout=10).json())

                for _, r in dataset.iterrows():
                    file, prompt = r["file"], r["prompt"]
                    print(f"\n{file} | {len(prompt)} chars | {len(prompt.split())} words")
                    try:
                        testo, tempo, prompt_tokens, completion_tokens = chiama_modello_testo(prompt)
                        print("   risposta:", testo)

                        #out = results_dir / f"{modello['label']}_{Path(file).stem}.txt"
                        #out.write_text(testo, encoding="utf-8")

                        writer.writerow({
                            "timestamp": datetime.now().isoformat(),
                            "modello": modello["label"],
                            "file": file,
                            "prompt": prompt,
                            "risposta": testo,
                            "num_parole": len(testo.split()),
                            "tempo_s": round(tempo, 2),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        })
                        f_csv.flush()
                    except Exception as e:
                        print("   ERRORE:", e)
            finally:
                ferma_server(processo)

    print(f"\nFatto. Risultati in: {csv_path.resolve()}")


main()