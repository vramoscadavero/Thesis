import subprocess
from datetime import datetime
import requests
from pathlib import Path
import time
import os


LLAMA_BIN = str(Path.home() / ".local" / "bin" / "llama")
GPU_ID = "0"

def avvia_server(repo, quant, output_dir):
    comando = [LLAMA_BIN, "serve", "-hf", f"{repo}:{quant}",
               "--temp", "0", "--top-k", "1",
               "-c", "8192", "-np", "1", 
               "--host", "127.0.0.1", "--port", "18080"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    log_path = output_dir / f"log_{repo.replace('/', '_')}.txt"
    log_file = open(log_path, "w")
    return subprocess.Popen(comando, stdout=log_file, stderr=subprocess.STDOUT, env=env)


def wait_server(timeout_s, SERVER_URL, processo=None):
    start = time.time()
    ultimo_msg = 0
    while time.time() - start < timeout_s:
        if processo is not None and processo.poll() is not None:
            print(f"Server terminato con codice {processo.returncode}")
            return False
        try:
            if requests.get(SERVER_URL + "/v1/models", timeout=3).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if time.time() - ultimo_msg > 30:
                print("In attesa del server...")
                ultimo_msg = time.time()
        time.sleep(2)
    return False


def ferma_server(processo):
    processo.terminate()
    try:
        processo.wait(timeout=15)
    except subprocess.TimeoutExpired:
        processo.kill()
        processo.wait()
    time.sleep(8)