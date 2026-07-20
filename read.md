
docker run --rm --gpus=all -v "%cd%:/app" -v zonos_cache:/root/.cache/huggingface zonos bash -c "python sample.py"