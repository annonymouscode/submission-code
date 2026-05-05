## Project Structure

```
submission-code/
├── run.py                          # Main training entry point
├── test_run.py                     # Evaluation script (loads pretrained checkpoint)
├── explainability.py               # Qualitative analysis / explainability
├── config.yaml                     # All hyperparameters and settings
├── constants.py                    # Paths, label mappings, field names
├── enums.py                        # Enumerations
│
├── models/
│   ├── cirsis_mmd_vision_text_pipeline/
│   │   └── model.py                # CAFuNet main model
│   └── vision_text_pipeline_common/
│       ├── clip_embedding.py       # CLIP encoder wrapper
│       ├── text_vision_fuser/
│       │   └── model.py            # Vision & Text encoders with topic conditioning
│       ├── block_fusion.py         # Projected Bilinear Block Fusion (PBBF)
│       ├── fuzz_feature_extractor.py       # Fuzzy feature extraction
│       └── fuzzy_membership_network.py     # CGC membership functions
│
├── topic_modelling/
│   ├── crisis_mmd.py               # BERTopic induction for CrisisMMD
│   └── tseqd.py                    # BERTopic induction for TSEqD
│
├── data/
│   ├── dataloader_interface.py     # DataLoader construction
│   ├── dataset_interface.py        # Dataset base interface
│   └── datasets/
│       ├── crisis_mmd_dataset/
│       │   └── dataset.py          # CrisisMMD data loading & preprocessing
│       └── tseqd_dataset/
│           └── dataset.py          # TSEqD data loading & preprocessing
│
├── model_meta_components/
│   └── loss_functions/
│       ├── vision_text_pipeline_loss_function.py   # CE + contrastive loss
│       └── vtp_cross_entropy_loss.py
│
├── runner_builders/
│   ├── base.py                     # Runner factory
│   └── crisis_mmd_runners.py       # Runner configs (optimizer, scheduler, weights)
│
├── runner_interfaces/
│   ├── pytorch_runner_interface.py  # Train/val/test epoch logic
│   └── ...
│
├── utils/
│   ├── config_utils.py             # YAML config loader
│   ├── metric_utils.py             # Accuracy, Precision, Recall, F1
│   ├── data_utils.py               # DataLoader param helpers
│   └── gpu_utils.py                # Device selection
│
└── requirements.txt
```

## Topic Induction

CAFuNet uses **BERTopic** with `distilroberta-base-msmarco-v1` sentence embeddings to extract domain-specific topics from the training corpus. Topics are induced offline and loaded during training.

To run topic induction for CrisisMMD:

```bash
python -m topic_modelling.crisis_mmd
```

For TSEqD:

```bash
python -m topic_modelling.tseqd
```

The induced topic models are saved to `topic_modelling/crisis_mmd_topic_model/` and `topic_modelling/tseqd_topic_model/`, respectively. Set `train_bert_topic: true` in `config.yaml` to re-train the topic model during the main run, or `false` to load from disk.

## Training

All training configuration is managed through `config.yaml`. Key hyperparameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `seed` | 11 | Random seed |
| `num_epochs` | 100 | Maximum training epochs (early stopping applied) |
| `train_batch_size` | 32 | Training batch size |
| `embedding_dim` | 512 | Shared embedding dimensionality |
| `num_selected_topics` | 40 | Number of topic embeddings (N_t) |
| `cl_loss_temperature` | 1.0 | Contrastive loss temperature (tau) |
| `max_len_text` | 128 | Maximum text token length |
| `use_lr_scheduler` | 1 | Enable StepLR scheduler |
| `apply_gradient_clipping` | 1 | Enable gradient clipping |
| `crisis_mmd_like_dataset_to_use` | `crisis_mmd_dataset` | Dataset selector (`crisis_mmd_dataset` or `tseqd_dataset`) |

**Optimizer:** AdamW (lr=5e-5, weight_decay=7e-5) with StepLR (step_size=5, gamma=0.8).

### Run training

```bash
# Single GPU
python run.py

# Distributed (multi-GPU) — set run_parallel: true in config.yaml
python run.py
```

To switch between datasets, update `crisis_mmd_like_dataset_to_use` in `config.yaml`:

```yaml
# For CrisisMMD
crisis_mmd_like_dataset_to_use: crisis_mmd_dataset

# For TSEqD
crisis_mmd_like_dataset_to_use: tseqd_dataset
```

## Evaluation

To evaluate a trained checkpoint on the test set:

```bash
python test_run.py
```
