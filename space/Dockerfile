# Hugging Face Docker Space: the planner app plus Ollama serving a small model on the free CPU.
FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# The model is baked into the image so a restart does not download it again.
ARG MODEL=granite4.2:3b
ENV OLLAMA_MODELS=/opt/ollama/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_CONTEXT_LENGTH=8192 \
    OLLAMA_KEEP_ALIVE=-1 \
    SPACE_MODEL=${MODEL}
RUN mkdir -p "$OLLAMA_MODELS" \
    && (ollama serve & sleep 5 && ollama pull "$MODEL") \
    && chmod -R a+rX /opt/ollama

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir .
COPY --chown=user start.sh ./

USER user
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["bash", "start.sh"]
