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
│   │   └── model.py                # main model
│   └── vision_text_pipeline_common/
│       ├── clip_embedding.py       
│       ├── text_vision_fuser/
│       │   └── model.py            
│       ├── block_fusion.py         
│       ├── fuzz_feature_extractor.py       
│       └── fuzzy_membership_network.py    
│
├── topic_modelling/
│   ├── crisis_mmd.py              
│   └── tseqd.py                   
│
├── data/
│   ├── dataloader_interface.py     # DataLoader construction
│   ├── dataset_interface.py        # Dataset base interface
│   └── datasets/
│       ├── crisis_mmd_dataset/
│       │   └── dataset.py          
│       └── tseqd_dataset/
│           └── dataset.py          
│
├── model_meta_components/
│   └── loss_functions/
│       ├── vision_text_pipeline_loss_function.py   
│       └── vtp_cross_entropy_loss.py
│
├── runner_builders/
│   ├── base.py                    
│   └── crisis_mmd_runners.py       
│
├── runner_interfaces/
│   ├── pytorch_runner_interface.py  
│   └── ...
│
├── utils/
│   ├── config_utils.py             # YAML config loader
│   ├── metric_utils.py             
│   ├── data_utils.py               
│   └── gpu_utils.py                
│
└── requirements.txt
```

## Topic Induction

```bash
python -m topic_modelling.crisis_mmd
```

For TSEqD:

```bash
python -m topic_modelling.tseqd
```

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
