# Sentiment Analysis with DeBERTa-v3

Use `microsoft/deberta-v3-large` to classify text sentiment with a simple training script.

## 1. Setup
1. `python3 -m venv venv`
2. `source venv/bin/activate` &nbsp;&nbsp;# Windows: `venv\Scripts\activate`
3. `pip install -r requirements.txt`

Python 3.8+ is required. A CUDA GPU is recommended

## 2. Data
- Place `train.csv` and `test.csv` in your working directory.
- Required columns: `text` (input string) and `label` (integer class).
- The script automatically holds out 5% of `train.csv` for validation, `val.csv` will saved in saved_models/auto_val_split.

## 3. Configure & Run
1. Edit `main.py` and set the paths in the `config` dict:
   - `train_csv`, `test_csv`, `out_dir`, etc.
2. Start training: `python main.py`

Key defaults: backbone `microsoft/deberta-v3-large`, batch size 32, max length 90, 10 epochs, AdamW with encoder LR `3e-5` and head LR `2e-4`, dropout 0.3, label smoothing 0.07.

## 4. Outputs
Results are saved under `out_dir`:
- `checkpoint/` with the best model and tokenizer.
- `training_history.png` for loss/accuracy curves.
- `*_cm.png` confusion matrices for train/val/test.
- `summary.json` with metrics and config.
- `*_report.txt` classification reports (precision/recall/F1).
