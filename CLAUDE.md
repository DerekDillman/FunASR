# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FunASR is an industrial-grade speech recognition toolkit (Python package `funasr`) built by Alibaba Group. It provides a unified `AutoModel` API over 50+ models covering ASR, VAD, punctuation restoration, speaker diarization, emotion detection, and keyword spotting. Models are downloaded from ModelScope (default) or HuggingFace on first use.

Current version: see `funasr/version.txt`.

## Commands

### Install for development

```bash
pip install -e ./
```

Optional extras: `pip install -e ".[test]"` (pytest), `pip install -e ".[llm]"` (LLM-based models), `pip install -e ".[doc]"` (Sphinx docs), `pip install -e ".[all]"` (everything).

### Run tests

```bash
# All unit tests (no model downloads)
python tests/run_test.py

# Single test file
python -m pytest tests/test_auto_model.py -v

# Single test case
python -m pytest tests/test_auto_model.py::TestAutoModel::test_progress_callback_called -v

# Tests that require model downloads (slow)
python tests_models/run_all_tests.py
```

Tests use `unittest`. The `tests/` directory contains unit tests that do not require model downloads. `tests_models/` contains integration tests that download real models.

### Code formatting

Pre-commit enforces Black at 100-character line length:

```bash
pip install pre-commit
pre-commit install
black --line-length=100 funasr/
```

### Syntax check

```bash
python -m compileall funasr examples tests
```

### CLI

```bash
funasr audio.wav                          # transcribe with SenseVoice (default)
funasr audio.wav --model paraformer       # specific model
funasr audio.wav --format srt            # SRT subtitle output
funasr-server --device cuda --port 8000  # OpenAI-compatible API server
```

### Training

```bash
# Single GPU
funasr-train ++model=paraformer-zh ++train_data_set_list=data/train.jsonl ++valid_data_set_list=data/valid.jsonl ++output_dir=exp/

# Multi-GPU DeepSpeed (distributed)
funasr-train-ds ++model=paraformer-zh ++train_data_set_list=data/train.jsonl ++output_dir=exp/ deepspeed_config=examples/deepspeed_conf/ds_stage2.json

# Export to ONNX
funasr-export ++model=paraformer-zh ++type=onnx ++quantize=false
```

Training commands use Hydra configuration; all kwargs use `++key=value` syntax to override config.yaml fields loaded from the downloaded model.

## Architecture

### Core inference flow

```
funasr/__init__.py
  └── import_submodules()         # auto-imports all subpackages to trigger @tables.register decorators
  └── from auto.auto_model import AutoModel

AutoModel.__init__(**kwargs)
  ├── build_model(model=...)      # downloads from hub, reads config.yaml, instantiates via registry
  ├── build_model(vad_model=...)  # optional: VAD for long-audio segmentation
  ├── build_model(punc_model=...) # optional: punctuation restoration
  └── build_model(spk_model=...) # optional: speaker diarization + ClusterBackend

AutoModel.generate(input=...)
  ├── prepare_data_iterator()     # accepts wav path, URL, bytes, list, .scp/.jsonl manifests
  ├── vad_model.inference()       # splits audio into segments if vad_model set
  ├── model.inference()           # main ASR / task model
  ├── punc_model.inference()      # adds punctuation if punc_model set
  └── spk_model.inference()       # adds speaker labels if spk_model set
```

### Registry system (`funasr/register.py`)

All components (models, frontends, encoders, decoders, tokenizers, etc.) are registered via decorator on import:

```python
@tables.register("model_classes", "FsmnVADStreaming")
class FsmnVADStreaming(nn.Module):
    ...
```

`funasr/__init__.py` calls `import_submodules(__name__)` at startup, which walks all subpackages and triggers every `@tables.register` decorator. `AutoModel.build_model()` then looks up the class name from `config.yaml` (the `model:` key) in `tables.model_classes`.

The `tables` singleton (`RegisterTables`) holds dictionaries: `model_classes`, `frontend_classes`, `encoder_classes`, `decoder_classes`, `predictor_classes`, `tokenizer_classes`, `dataset_classes`, etc.

**To add a new model:** create a directory under `funasr/models/`, write `model.py` with `@tables.register("model_classes", "MyModelName")`, and provide a `template.yaml` showing the expected `config.yaml` shape. No manual registration list to update — the auto-import handles discovery.

### Model download and config (`funasr/download/`)

`download_model_from_hub.py` resolves short aliases (e.g. `"fsmn-vad"` → full ModelScope ID) via `name_maps_from_hub.py`, downloads model files, and reads `config.yaml`. The YAML drives instantiation — `model:`, `encoder:`, `decoder:`, `frontend:`, and `tokenizer:` keys name registered classes; `model_conf:`, `encoder_conf:`, etc. are kwargs passed to their constructors.

Hub selection: `hub="ms"` (ModelScope, default) or `hub="hf"` (HuggingFace). Disable version checks with `disable_update=True`.

### Model directory layout pattern

Each model in `funasr/models/<name>/` typically contains:
- `model.py` — the `nn.Module` with `@tables.register`, plus `inference()` and optionally `export()` methods
- `encoder.py`, `decoder.py`, etc. — subcomponents also registered with `@tables.register`
- `template.yaml` — reference config showing all supported config keys
- `export_meta.py` — ONNX/TorchScript export metadata (optional)

### Inference API shape

Every registered model must implement:
```python
def inference(self, data_in, input_len=None, key=None, tokenizer=None, frontend=None, **kwargs):
    ...
    return results, meta_data  # results: list of dicts with "text", "timestamp", etc.
```

### Training architecture (`funasr/train_utils/`)

- `trainer.py` — `Trainer` class: single-node DDP/FSDP training loop with mixed precision, TensorBoard logging, checkpoint save/resume
- `trainer_ds.py` — DeepSpeed-aware trainer for multi-node distributed training
- `load_pretrained_model.py` — loads `model.pt` into the instantiated model
- `initialize.py` — weight initialization utilities

Training entry points (`funasr/bin/train.py`, `train_ds.py`) use `@hydra.main` — all configuration is passed via Hydra `DictConfig`.

### Data pipeline (`funasr/datasets/`)

Dataset classes are also registered in `tables.dataset_classes`. Standard training data format is JSONL (`{"source": "/path/to/audio.wav", "target": "transcript text"}`). Helper CLIs: `funasr-scp2jsonl`, `funasr-jsonl2scp`, `funasr-sensevoice2jsonl`.

### vLLM inference (`funasr/auto/auto_model_vllm.py`)

`AutoModelVLLM` wraps LLM-based models (FunASRNano, LLMASR, GLMASR) for vLLM-accelerated batch inference. Not applicable to non-autoregressive models (Paraformer, SenseVoice, CT-Transformer, Qwen3-ASR). Requires `pip install vllm`.

### Server (`funasr/bin/server.py`, `funasr/bin/_server_app.py`)

`funasr-server` starts an OpenAI-compatible FastAPI server (`/v1/audio/transcriptions`). Requires `pip install fastapi uvicorn python-multipart`. Model aliases (`sensevoice`, `paraformer`, `fun-asr-nano`) are resolved in `funasr/cli.py`'s `MODEL_CONFIGS`.

## Key model aliases

Short names accepted by `AutoModel(model=...)` are resolved via `funasr/download/name_maps_from_hub.py`:

| Alias | Task |
|---|---|
| `fsmn-vad` | Voice Activity Detection |
| `ct-punc` | Punctuation restoration (CN+EN) |
| `cam++` | Speaker diarization |
| `paraformer-zh` | Mandarin ASR |
| `paraformer-en` | English ASR |
| `paraformer-zh-streaming` | Streaming Mandarin ASR |
| `iic/SenseVoiceSmall` | Multilingual ASR + emotion + audio events |
| `emotion2vec_plus_large` | Emotion recognition |

## Environment variables

| Variable | Effect |
|---|---|
| `FUNASR_IMPORT_DEBUG=1` | Print details on submodule import failures during startup |
| `FUNASR_STRICT_IMPORT=1` | Raise on any submodule import failure instead of silently skipping |
| `HYDRA_FULL_ERROR=1` | Show full Hydra tracebacks (set automatically by `funasr/__init__.py`) |

## CI/CD

GitHub Actions (`.github/workflows/update-api-docs.yml`) regenerates and publishes API docs to `gh-pages` on every push to `main` that touches `funasr/**/*.py`. There are no automated test runs in CI — tests must be run locally before submitting PRs.

## Development branch

Active development should target the branch `claude/claude-md-docs-cm1our` in the `derekdillman/funasr` fork.
