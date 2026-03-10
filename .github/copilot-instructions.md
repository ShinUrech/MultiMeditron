# MultiMeditron — Copilot Workspace Instructions

## Project Overview

MultiMeditron is a multimodal medical AI model developed at EPFL / LIGHТ lab. It integrates multiple specialist CLIP vision encoders with a large language model (LLaMA / Qwen / Apertus family) via a Mixture-of-Experts (MoE) image modality. The repo lives at `/users/surech/meditron/MultiMeditron` on the Clariden HPC cluster.

---

## Cluster Environment

| Item | Value |
|------|-------|
| Cluster | Clariden (CSCS) |
| SLURM account | `a127` |
| Container env | `/users/surech/.edf/multimeditron.toml` |
| Reports / logs | `/users/surech/meditron/reports/` |
| WandB dir | `/capstor/store/cscs/swissai/a127/homes/surech/wandb` (offline) |
| HF cache | `/capstor/store/cscs/swissai/a127/meditron/hf_cache` |

### Key storage paths on Clariden (`/capstor/store/cscs/swissai/a127/meditron/`)

```
models/
  openai/clip-vit-base-patch32/          # Generalist CLIP
  CLIP/OphthalmologyExpert/              # Eye specialist
  CLIP/SkinExpert/                       # Skin specialist
  CLIP/UltraSoundCLIP/checkpoint-4350/  # Ultrasound specialist
multimediset/arrow/
  llava_pretrain_cleaned/
  pixmo_anything/
  pixmo_cap/
  medtrinity_conversations_1_formatted_alignment/
  eye_dataset/
  skin_dataset/
```

---

## Repository Structure

```
src/multimeditron/
  model/
    model.py              # MultiModalModelForCausalLM, MultimodalConfig, ChatTemplate
    modalities/
      base.py             # BaseModality, BaseModalityProcessor, AutoModality registry
      image_modality.py   # Single CLIP (meditron_clip)
      image_modality_biomed.py  # BiomedCLIP/OpenCLIP (meditron_biomedclip)
      image_modality_moe.py     # MoE shared projector (moe_meditron_clip)
      image_modality_moe_pep.py # MoE per-expert projector (moe_meditron_clip_pep) ← active
      moe/gating.py       # GatingNetwork (ResNet50-based image router)
    projectors/mlp.py     # MLPProjector (3-layer MLP with GELU)
    attention.py          # CrossAttention (used for cross_attn fusion mode)
    data_loader.py        # DataCollatorForMultimodal
    constants.py          # Key string constants (MODALITY_VALUE_KEY etc.)
  train/trainer.py        # MultimodalTrainer (extends HF Trainer); TrainingMode enum
  cli/
    train.py              # `multimeditron train --config <yaml>` entry point
    experts.py            # `multimeditron train_expert` / `batch_train_expert`
  experts/
    train_clip.py         # Expert CLIP fine-tuning
    config_maker.py       # Generates YAML configs from dataset/hp sweep specs
  dataset/
    loader/               # AutoModalityLoader, raw-image loader
    sample_preprocessor.py
    preprocessor/
cookbook/sft/             # Training YAML configs, organized by variant
  moe/attn/pep/           # ← Active training configs (cross_attn + per-expert proj)
    stage1_alignment.yaml
    stage2_end2end.yaml
  moe/avg/…               # weighted_average fusion variant
  moe/cat/…               # sequence_append fusion variant
  single_clip/…           # Non-MoE single encoder variants
config/deepspeed.json     # DeepSpeed config used in training_args
sbatch_train.sh           # SLURM job script (torchrun, 4 GPUs, project a127)
scripts/
  prep_image_datasets.py  # Dataset preparation for eye/skin datasets
  train_clip.py           # Standalone expert training script
```

---

## Model Architecture

```
Input image → GatingNetwork (ResNet50) → expert weights
Each image → N expert CLIPs (vision_model only, CLS token dropped)
Per-expert MLPProjector → projected tokens
Fusion method (weighted_average | cross_attn | sequence_append)
→ LLM embedding space → LLaMA/Qwen/Apertus LLM
```

### Registered modality types (via `@AutoModality.register`)

| `model_type` | Class | Description |
|---|---|---|
| `meditron_clip` | `ImageModality` | Single HF CLIP |
| `meditron_biomedclip` | `BioMedCLIPImageModality` | OpenCLIP / BiomedCLIP |
| `moe_meditron_clip` | `MOEImageModality` | MoE with shared projector |
| `moe_meditron_clip_pep` | `MOEImageModalityPEP` | MoE with per-expert projector ← default |

### Training modes (`TrainingMode` enum)

| Mode | LLM | CLIP encoders | Projectors |
|---|---|---|---|
| `ALIGNMENT` | frozen | frozen | trainable |
| `END2END` | trainable | frozen | trainable |
| `LM_ONLY` | trainable | frozen | frozen |
| `FULL` | trainable | trainable | trainable |

### Chat templates

`ChatTemplate.from_name()` supports: `"llama"`, `"apertus"`, `"qwen3"`.  
Attachment placeholder token: `<|reserved_special_token_0|>`.

---

## Training Pipeline

### Launching a training job

```bash
cd /users/surech/meditron/MultiMeditron
sbatch sbatch_train.sh
```

The script uses `torchrun` with 4 GPUs on 1 node, runs inside the `multimeditron.toml` container, and calls `python -m multimeditron train --config $CONFIG`.

### YAML config structure (see `cookbook/sft/moe/attn/pep/stage1_alignment.yaml`)

```yaml
base_llm: meta-llama/Llama-3.1-8B-Instruct
tokenizer_type: llama           # or apertus, qwen3
attachment_token: <|reserved_special_token_0|>
token_size: 4096

loaders:
  - loader_type: raw-image
    modality_type: image

modalities:
  - model_type: moe_meditron_clip_pep
    image_processor: <path to clip for preprocessing>
    hidden_size: 4096
    expert_clip_names: [...]
    generalist_idx: 0           # index into expert_clip_names
    gating_path: ""             # empty = random init
    fusion_method: cross_attn   # or weighted_average, sequence_append
    top_k_experts: 3

training_mode: ALIGNMENT
datasets:
  - packed_path: <HF arrow dataset path>
training_args:
  output_dir: ...
  deepspeed: /users/surech/meditron/MultiMeditron/config/deepspeed.json
  ...
```

### Monitoring

```bash
squeue --me                               # check running jobs
tail -f /users/surech/meditron/reports/R-multimeditron-stage1.<jobid>.out
tail -f /users/surech/meditron/reports/R-multimeditron-stage1.<jobid>.err
```

---

## Known Issues / Important Notes

- **`kwargs=kwargs` bug**: All four image config `__init__` methods pass `kwargs=kwargs` instead of `**kwargs` to `super().__init__()`. Extra config fields (e.g. HF `name_or_path`) are silently swallowed. Not yet fixed.
- **`freeze_modality_embedder` not abstract**: In `BaseModality`, this method is missing `@abstractmethod` — subclasses that forget to implement it will silently do nothing when frozen.
- **`MOEImageConfig` / `MOEImageConfigPEP` are duplicated**: ~9 identical fields. A shared `BaseMOEImageConfig` parent has been identified but not yet created.
- **`gating_path: ""`**: When empty, the gating network is randomly initialized. This is intentional for training from scratch; provide a checkpoint path to resume.
- **WandB is offline** on Clariden. Sync manually with `wandb sync <run_dir>` after the job.
- **Datasets** must be in HuggingFace Arrow format (`load_from_disk`-compatible) at the paths listed in the YAML. The `packed_path` is detected as a folder dataset if it contains `dataset_info.json` and `state.json`.

---

## CLI Entry Points

```bash
multimeditron train --config <yaml>          # main training
multimeditron train_expert <config.yaml>     # fine-tune a single CLIP expert
multimeditron batch_train_expert <c1> <c2>   # parallel expert training
multimeditron config_maker_expert <configs>  # generate expert training configs
multimeditron preprocess ...                 # dataset preprocessing
multimeditron check_dataset ...              # dataset validation
```

---

## Code Conventions

- All modalities registered via `@AutoModality.register("<name>")` decorator
- Config classes inherit from `PretrainedConfig` (HuggingFace standard)
- `BaseModality` inherits from both `ABC` and `PreTrainedModel`
- dtype is always `bfloat16` on Clariden H100s
- DeepSpeed ZeRO config at `config/deepspeed.json` is used for all multi-GPU runs
- Tests: `pytest tests/ -x -q`
