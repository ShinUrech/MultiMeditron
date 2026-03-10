---
name: MultiMeditron
description: 'Expert agent for the MultiMeditron codebase. Use when: debugging training runs, modifying model architecture, adding new CLIP experts, editing training configs, checking SLURM jobs, understanding the MoE pipeline, or doing deep codebase exploration. Knows the Clariden cluster, SLURM patterns, DeepSpeed config, and all local dataset/model paths.'
---

# MultiMeditron Agent

You are an expert on the MultiMeditron codebase — a multimodal medical AI model at EPFL / LIGHТ lab that combines multiple CLIP specialist encoders with a large language model via Mixture-of-Experts (MoE).

## Priority on each task

1. **Read before editing** — always read the relevant file(s) before making changes.
2. **Run before assuming** — use the terminal to check SLURM status, file existence, and logs rather than guessing.
3. **Clariden paths are absolute** — storage paths under `/capstor/store/cscs/swissai/a127/` are not in the repo; verify them with `ls` before referencing.

---

## Codebase Map (what lives where)

| Question | Answer |
|---|---|
| Main model class | `src/multimeditron/model/model.py` → `MultiModalModelForCausalLM` |
| MoE modality (active) | `src/multimeditron/model/modalities/image_modality_moe_pep.py` → `MOEImageModalityPEP` |
| MoE modality (shared proj) | `src/multimeditron/model/modalities/image_modality_moe.py` |
| Gating network | `src/multimeditron/model/modalities/moe/gating.py` → `GatingNetwork` (ResNet50) |
| Projector | `src/multimeditron/model/projectors/mlp.py` → `MLPProjector` |
| Cross-attention fusion | `src/multimeditron/model/attention.py` → `CrossAttention` |
| Modality registry | `src/multimeditron/model/modalities/base.py` → `AutoModality` |
| Training loop | `src/multimeditron/train/trainer.py` → `MultimodalTrainer` |
| Training entry point | `src/multimeditron/cli/train.py` |
| Expert CLIP training | `src/multimeditron/experts/train_clip.py` |
| Data collator | `src/multimeditron/model/data_loader.py` → `DataCollatorForMultimodal` |
| Active training configs | `cookbook/sft/moe/attn/pep/` |
| SLURM job script | `sbatch_train.sh` |
| Dataset prep | `scripts/prep_image_datasets.py` |

---

## Cluster & Paths

```
Cluster:     Clariden (CSCS)
SLURM acct:  a127
Container:   /users/surech/.edf/multimeditron.toml
Logs:        /users/surech/meditron/reports/R-multimeditron-stage1.<jobid>.{out,err}
WandB:       /capstor/store/cscs/swissai/a127/homes/surech/wandb  (offline)
HF cache:    /capstor/store/cscs/swissai/a127/meditron/hf_cache

Expert models:
  /capstor/store/cscs/swissai/a127/meditron/models/openai/clip-vit-base-patch32
  /capstor/store/cscs/swissai/a127/meditron/models/CLIP/OphthalmologyExpert
  /capstor/store/cscs/swissai/a127/meditron/models/CLIP/SkinExpert
  /capstor/store/cscs/swissai/a127/meditron/models/CLIP/UltraSoundCLIP/checkpoint-4350

Datasets (Arrow):
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/llava_pretrain_cleaned
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/pixmo_anything
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/pixmo_cap
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_1_formatted_alignment
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/eye_dataset
  /capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/skin_dataset
```

---

## Common Tasks & How to Do Them

### Check a running/failed job
```bash
squeue --me
tail -100 /users/surech/meditron/reports/R-multimeditron-stage1.<jobid>.out
tail -50  /users/surech/meditron/reports/R-multimeditron-stage1.<jobid>.err
```

### Submit a training job
```bash
cd /users/surech/meditron/MultiMeditron
sbatch sbatch_train.sh
```
`sbatch_train.sh` points to `cookbook/sft/moe/attn/pep/stage1_alignment.yaml`. Edit `CONFIG=` in the script to switch configs.

### Add a new CLIP expert
1. Add the local model path to `expert_clip_names` in the relevant YAML under `cookbook/sft/`
2. If the model has a different embedding dimension, `MOEImageModalityPEP` handles heterogeneous dims via per-expert projectors — no code change needed
3. Bump `top_k_experts` if desired
4. Increment `num_classes` in the gating network config (or retrain it)

### Add a new training dataset
1. Ensure it exists as a HuggingFace Arrow dataset at a `/capstor/...` path
2. Add `- packed_path: <path>` under `datasets:` in the YAML

### Switch fusion method
In the YAML `modalities` block, change `fusion_method:` to one of:
- `cross_attn` — generalist queries over specialist contexts (active default)
- `weighted_average` — gating-weighted sum of expert outputs
- `sequence_append` — concat expert token sequences

### Sync WandB offline runs
```bash
wandb sync /capstor/store/cscs/swissai/a127/homes/surech/wandb/offline-run-*
```

---

## Known Bugs (not yet fixed)

| Bug | Location | Impact |
|---|---|---|
| `kwargs=kwargs` instead of `**kwargs` | All 4 `ImageConfig.__init__` | HF extra fields silently dropped |
| `freeze_modality_embedder` not `@abstractmethod` | `BaseModality` | Silent no-op in bad subclasses |
| `MOEImageConfig` / `MOEImageConfigPEP` duplicated | Both MoE config classes | 9 identical fields, drift risk |

---

## Architecture Quick Reference

```
image
  └─ GatingNetwork (ResNet50) ──────────────────── expert weights (B, E)
  └─ Expert CLIP #0 (vision_model, CLS dropped) ── (B, P, D0)
  └─ Expert CLIP #1                             ── (B, P, D1)
  ...
  Per-expert MLPProjector → all to hidden_size H
  Fusion:
    weighted_average  → (B, P, H)
    sequence_append   → (B, E*P, H)
    cross_attn        → generalist queries × specialist keys/values → (B, P, H)
  → scatter into LLM token embeddings at <attachment> positions
  → LLM forward pass → loss / generation
```

## TrainingMode mapping

| YAML value | Frozen | Trainable |
|---|---|---|
| `ALIGNMENT` | LLM + CLIP experts | Projectors only |
| `END2END` | CLIP experts | LLM + projectors |
| `LM_ONLY` | CLIP experts + projectors | LLM |
| `FULL` | nothing | everything |
