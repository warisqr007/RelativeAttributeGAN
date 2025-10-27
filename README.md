# Relative Attribute GAN for Speaker Embeddings

## Project Structure

```
RelativeAttributeGAN/
├── models/
│   ├── __init__.py
│   ├── speaker_encoder.py      # Pre-trained speaker embedding extractor
│   ├── attribute_ranker.py     # Pairwise attribute ranking networks
│   ├── generator.py            # GAN generator
│   ├── discriminator.py        # Multi-task discriminator
│   └── losses.py               # All loss functions
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset classes
│   └── preprocessing.py        # Data preparation utilities
├── training/
│   ├── __init__.py
│   ├── ranker_trainer.py       # Phase 1: Train attribute rankers
│   ├── gan_trainer.py          # Phase 2-3: GAN training
│   └── curriculum.py           # Progressive training scheduler
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # EER, ranking accuracy, etc.
│   └── evaluator.py            # Evaluation pipeline
├── configs/
│   ├── ranker_config.yaml
│   └── gan_config.yaml
└── main.py
```

## Installation Requirements

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install speechbrain  # For ECAPA-TDNN
pip install librosa soundfile
pip install scikit-learn scipy
pip install wandb  # For logging
pip install torchmetrics
pip install omegaconf pyyaml
```

---