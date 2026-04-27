# Imagen base oficial de vLLM con soporte para Qwen3.5-MoE
FROM vllm/vllm-openai:latest

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Instalar runpod SDK para el handler serverless
RUN pip install --no-cache-dir runpod hf-transfer

# Copiar el handler
COPY handler.py /handler.py

# Override del entrypoint de vLLM para usar nuestro handler
ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]