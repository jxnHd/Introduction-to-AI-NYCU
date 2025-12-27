HW4 Object Detection (Pig Detection)Student ID: 111511281本專案使用 YOLOv11 Large (yolo11l.pt) 進行豬隻偵測 (Pig Detection)。This repository contains scripts for training the model and evaluating it on a test dataset.1. Environment Requirements (環境需求)請確保您的環境已安裝以下套件：To run the code, ensure the following dependencies are installed:Python 3.8+PyTorch (with CUDA or MPS support recommended)Ultralytics YOLOpip install ultralytics torch pyyaml
2. Project Structure (檔案結構)執行程式前，請確認檔案結構如下：Ensure the directory structure matches the following setup:.
├── train.py               # 訓練用程式碼 (Training Script)
├── test.py                # 測試與評估程式碼 (Evaluation Script)
├── 111511281.pt       # 預訓練權重檔 (Pre-trained weights, required for testing)
├── obj_data/              # 資料集根目錄 (Dataset root)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── ...
└── README.md
3. How to Run Training (如何執行訓練)訓練程式 (train.py) 使用 yolo11l.pt 作為基礎模型，並包含資料增強設定 (Mosaic, Mixup, HSV 等)。Steps:將訓練資料放入 obj_data 資料夾。執行訓練腳本：python train.py
Output (輸出):訓練過程與 Log 會儲存在 runs/detect/HW4_111511281/。訓練完成後，最佳權重 (Best weights) 會自動複製到當前目錄並命名為 111511281.pt。4. How to Run Evaluation (如何執行評估)test.py 負責執行以下兩項工作：Validation: 計算指標 (mAP@50, mAP@50-95) 並產生 Confusion Matrix 與 PR Curve。Prediction: 對測試集圖片進行推論並畫框 (Visualized bounding boxes)。For TA: Private Dataset Configuration (助教 Private Set 設定說明)本程式設計為會自動根據資料集路徑產生對應的 .yaml 檔，您不需要手動製作 YAML。The script automatically generates the required .yaml file based on the dataset root path.如果您要使用 Private Set 進行測試，請依下列步驟修改：開啟 test.py。找到 Line 13 的 DATASET_ROOT 變數。將其修改為您的 Private Set 資料夾路徑。# Inside test.py (Line 13)

# ==========================================
# 設定區
# ==========================================
STUDENT_ID = "111511281"
MODEL_PATH = f"{STUDENT_ID}.pt" 

# [IMPORTANT] Modify this path to your private dataset root
# 請修改此處路徑指向您的 Private Set
DATASET_ROOT = "/path/to/your/private_dataset" 
Note: 程式會預期您的資料夾內包含 test/images 子目錄 (例如：/path/to/private_dataset/test/images)。Execution (執行指令)設定好路徑後（或使用預設的 obj_data），直接執行：python test.py
5. Outputs & Results (輸出結果)執行 test.py 後，結果將儲存於以下路徑：Evaluation Metrics (評估圖表):包含 Confusion Matrix, PR Curve 等。Path: runs/detect/test_evaluation/Visualized Predictions (預測結果圖片):包含畫上 Bounding Box 的測試集圖片。Path: runs/detect/test_prediction/6. Special Notes (注意事項)Weight File Name: test.py 預設讀取的權重檔名為 111511281_v18.pt。如果您是剛重新訓練完模型，請將產出的 111511281.pt 改名，或直接修改 test.py 中的 MODEL_PATH 變數。Device: 程式會自動偵測並使用 CUDA (NVIDIA) 或 MPS (Apple Silicon) 加速；若無則使用 CPU。