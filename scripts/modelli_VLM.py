import base64
import csv
import time
import sys
from pathlib import Path
from datetime import datetime
import requests
from llama_server import avvia_server, wait_server, ferma_server


models = [{
        "label": "Qwen3-VL-8B",
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "quant": "Q4_K_M"
      }, {
        "label": "Qwen3-VL-4B",
        "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "quant": "Q4_K_M"
      }, {
        "label": "Gemma-3-4B",
        "repo": "ggml-org/gemma-3-4b-it-GGUF",
        "quant": "Q4_K_M"
      },    {
        "label": "Gemma-4-E4B",
        "repo": "ggml-org/gemma-4-E4B-it-GGUF",
        "quant": "Q4_0"
    }]

image_dir = Path("./data")
output_dir = Path("./outputs/VLM")

general_prompt = 'Describe this ID image within 40 words'

specific_prompt = (
    'Describe this ID photo in maximum 40 words, as one grammatically connected '
    'sentence using conjunctions (e.g. "with", "and", "while") — not a list of '
    'short fragments separated by periods. Cover exactly these six attributes, '
    'in this priority order (most important first): eye position and openness, '
    'gaze direction, headwear presence or absence, background uniformity, '
    'lighting evenness, facial expression neutrality. Always explicitly state '
    'each attribute, even when the answer is negative or absent (e.g. '
    '"no headwear", "not smiling"). Do not mention any other attribute '
    '(e.g. glasses, makeup, hairstyle, hair color, jewelry, insignia, clothing '
    'details, accessories). Do not include subjective or aesthetic comments '
    '(e.g. "attractive", "nice photo"). Do not include summary or evaluative '
    'statements about overall compliance (e.g. "meeting all required attributes", '
    '"without deviation"). Only describe the six attributes themselves — do not '
    'judge or summarize their conformity as a whole. Do not repeat information. '
    'Be factual and concise. Maximum 40 words — this is a hard constraint.'
)

SERVER_URL = "http://127.0.0.1:18080"


def immagine_a_base64(path):
    with open(path, "rb") as f:
        dati = f.read()
    return base64.b64encode(dati).decode("utf-8")


def chiama_modello(prompt, immagine_path):
    """Manda una richiesta di chat completion con immagine al server locale.
    Ritorna un dizionario con testo generato, tempo, token, ecc."""
    img_b64 = immagine_a_base64(immagine_path)
    ext = immagine_path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                ],
            }
        ],
        "temperature": 0,
        "top_k": 1,
    }

    t0 = time.time()
    response = requests.post(SERVER_URL + "/v1/chat/completions", json=payload, timeout=240)
    elapsed = time.time() - t0
    response.raise_for_status()
    data = response.json()

    testo = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})

    return {
        "testo": testo,
        "tempo_s": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "num_parole": len(testo.split()),
    }


def testa_modello(modello, immagini, writer):
    """Esegue il loop su tutte le immagini per un singolo modello già caricato."""
    for img_path in immagini:
        print(f"    -> {img_path.name} ...", end=" ", flush=True)
        try:
            risultato = chiama_modello(general_prompt, img_path)
            print(f"OK ({risultato['tempo_s']}s, {risultato['num_parole']} parole)")
        except Exception as e:
            print(f"ERRORE: {e}")
            risultato = {
                "testo": f"[ERRORE: {e}]",
                "tempo_s": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "num_parole": None,
            }

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modello": modello["label"],
            "repo": modello["repo"],
            "quant": modello["quant"],
            "immagine": img_path.name,
            "risposta": risultato["testo"],
            "num_parole": risultato["num_parole"],
            "tempo_s": risultato["tempo_s"],
            "prompt_tokens": risultato["prompt_tokens"],
            "completion_tokens": risultato["completion_tokens"],
        })


def main():
    output_dir.mkdir(exist_ok=True, parents=True)

    immagini = sorted(
        [p for p in image_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not immagini:
        print(f"Nessuna immagine trovata in {image_dir.resolve()}")
        sys.exit(1)
    print(f"Trovate {len(immagini)} immagini in {image_dir}")

    csv_path = output_dir / f"risultati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    campi = ["timestamp", "modello", "repo", "quant", "immagine", "risposta",
             "num_parole", "tempo_s", "prompt_tokens", "completion_tokens"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campi)
        writer.writeheader()

        for modello in models:
            print(f"\n=== Modello: {modello['label']} ({modello['repo']}:{modello['quant']}) ===")
            processo = avvia_server(modello["repo"], modello["quant"], output_dir)
            try:
                print("  In attesa che il server sia pronto (download + caricamento)...")
                if not wait_server(900, SERVER_URL):
                    print("  ERRORE: il server non ha risposto in tempo, salto questo modello.")
                    continue
                print("  Server pronto. Inizio test sulle immagini...")
                testa_modello(modello, immagini, writer)
                f.flush()
            finally:
                print("  Chiudo il server...")
                ferma_server(processo)

    print(f"\nFatto. Risultati salvati in: {csv_path.resolve()}")


if __name__ == "__main__":
    main()