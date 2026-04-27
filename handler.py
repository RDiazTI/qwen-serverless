"""
Handler de RunPod Serverless para Qwen3.5-35B-A3B con vLLM.
Inicia un servidor vLLM en segundo plano y reenvía requests de RunPod a vLLM.
"""
import os
import time
import subprocess
import threading
import requests
import runpod

# ====================================================================
# CONFIGURACIÓN — todo se lee de variables de entorno
# ====================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "32768")
QUANTIZATION = os.getenv("QUANTIZATION", "moe_wna16")
DTYPE = os.getenv("DTYPE", "bfloat16")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "fp8")
GPU_MEM_UTIL = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
REASONING_PARSER = os.getenv("REASONING_PARSER", "qwen3")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

# ====================================================================
# ARRANQUE DE vLLM EN SEGUNDO PLANO
# ====================================================================
def start_vllm():
    """Lanza vLLM como subprocess."""
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--quantization", QUANTIZATION,
        "--dtype", DTYPE,
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--reasoning-parser", REASONING_PARSER,
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]
    print(f"[handler] Lanzando vLLM: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd)

def wait_for_vllm(timeout=900):
    """Espera hasta que vLLM esté listo (max 15 min)."""
    print(f"[handler] Esperando que vLLM levante en {VLLM_BASE}...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_BASE}/models", timeout=5)
            if r.status_code == 200:
                print(f"[handler] vLLM listo en {int(time.time()-start)}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM no levantó en {timeout}s")

# Lanzar vLLM una sola vez al arrancar el worker
start_vllm()
wait_for_vllm()

# ====================================================================
# HANDLER QUE RUNPOD LLAMA POR CADA REQUEST
# ====================================================================
def handler(event):
    """
    Recibe el evento de RunPod, reenvía a vLLM, devuelve la respuesta.
    
    Formato de entrada esperado:
    {
        "input": {
            "messages": [{"role": "user", "content": "..."}],
            "max_tokens": 100,
            "temperature": 0.7
        }
    }
    """
    try:
        body = event.get("input", {})
        
        # Asegurar que el modelo está incluido
        if "model" not in body:
            body["model"] = MODEL_NAME
        
        # Reenviar a vLLM
        r = requests.post(
            f"{VLLM_BASE}/chat/completions",
            json=body,
            timeout=600,
        )
        
        if r.status_code != 200:
            return {"error": f"vLLM returned {r.status_code}", "details": r.text}
        
        return r.json()
    
    except Exception as e:
        return {"error": str(e)}

# ====================================================================
# ARRANCAR EL WORKER DE RUNPOD
# ====================================================================
runpod.serverless.start({"handler": handler})