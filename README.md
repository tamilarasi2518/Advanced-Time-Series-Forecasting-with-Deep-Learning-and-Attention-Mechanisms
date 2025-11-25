# Advanced Time Series Forecasting - Fixed Project
This repository contains a corrected and working reference implementation for the "Advanced Time Series Forecasting with Deep Learning and Attention" project.

## What I fixed
- Implemented dataset acquisition (synthetic generator + CSV loader) and preprocessing with scaling and sequence creation.
- Implemented a working AttentionLSTM PyTorch model integrated with training and inference code.
- Added a simple standard LSTM baseline and a SARIMAX wrapper (if statsmodels installed).
- Added evaluation metrics (RMSE, MAE) and visualization helpers for predictions and attention weights.
- Added a minimal training script demonstrating a full train/val loop with checkpointing.

## Files
- `data.py` - data utilities (generate, load, scale, sequence)
- `model.py` - AttentionLSTM implementation
- `train.py` - training loop and demo
- `baselines.py` - Simple LSTM baseline and SARIMAX wrapper
- `evaluate.py` - metrics
- `visualize.py` - plotting helpers

## Quick start (local)
1. Install dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib
   # statsmodels optional for SARIMAX baseline
   pip install statsmodels
   ```
2. Run training demo (uses synthetic data):
   ```bash
   python train.py
   ```
3. To use your CSV dataset, modify `train.py` to call `load_csv` from `data.py` and then `scale_split`.

## Notes for submission
- Replace the synthetic generator with the Electricity Consumption dataset (if provided).
- Include plots of attention weights and comparative metrics (RMSE/MAE) between AttentionLSTM, SimpleLSTM and SARIMAX in the final report.
- Update README with dataset description, experiments, hyperparameters, and results.
