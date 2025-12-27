import os
import yaml
from ultralytics import YOLO
import torch

# ==========================================
# 設定區
# ==========================================
STUDENT_ID = "111511281"     # 請修改
MODEL_PATH = f"{STUDENT_ID}_v18.pt"  # 權重檔

# 資料路徑設定
DATASET_ROOT = "./obj_data"
DATA_YAML = os.path.join(DATASET_ROOT, "data_inplace.yaml")
TEST_DIR_SOURCE = os.path.join(DATASET_ROOT, "test", "images") # 測試集圖片路徑

def check_and_create_yaml(root_dir, yaml_path):
    if os.path.exists(yaml_path):
        print(f"[檢查] YAML 設定檔已存在: {yaml_path}")
        return

    print(f"[警告] 找不到 {yaml_path}，正在自動重新建立...")

    # 建立基本的 YAML 內容
    # 注意：這裡預設 train/val 資料夾結構存在於 root_dir 下
    # 即使 train/val 資料夾是空的，只要 'test' 路徑正確，model.val(split='test') 就能運作
    yaml_content = {
        'path': os.path.abspath(root_dir),
        'train': 'train/images',
        'val': 'test/images', # 若沒有 val 資料夾，暫時指向 train 避免報錯
        'test': 'test/images',
        'names': {0: 'pig'}
    }

    # 嘗試偵測真實的 val 資料夾
    if os.path.exists(os.path.join(root_dir, 'val', 'images')):
        yaml_content['val'] = 'val/images'

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        print(f"[成功] YAML 設定檔已建立: {yaml_path}")
    except Exception as e:
        print(f"[錯誤] 無法建立 YAML 檔: {e}")

def run_test():
    # 1. 檢查權重
    if not os.path.exists(MODEL_PATH):
        # 嘗試找 runs 裡的備份
        backup = f"runs/detect/HW4_{STUDENT_ID}/weights/best.pt"
        if os.path.exists(backup):
            model_path = backup
            print(f"使用備份權重: {model_path}")
        else:
            print(f"[錯誤] 找不到權重檔: {MODEL_PATH}")
            print(f"請確認是否已執行 train.py 並成功匯出模型。")
            return
    else:
        model_path = MODEL_PATH

    print(f"載入模型: {model_path}")
    model = YOLO(model_path)
    # 2. 產出評估圖表 (Confusion Matrix, PR Curve...)
    print("\n[Step 1] 正在產生評估圖表...")

    # --- 新增：自動檢查並建立 YAML ---
    check_and_create_yaml(DATASET_ROOT, DATA_YAML)
    # --------------------------------

    if os.path.exists(DATA_YAML):
        try:
            metrics = model.val(
                data=DATA_YAML,
                split='test',    # 指定使用 YAML 中的 test 資料集
                imgsz=640,       # 配合 train.py 的設定 (若您之前改成 640 請在此調整)
                batch=16,
                conf=0.001,      # 低信心度以繪製完整曲線
                iou=0.5,         # 設定 NMS IoU 為 0.5
                augment = True,    # 開啟 TTA
                plots=True,      # 畫出圖表
                device=('mps' if torch.backends.mps.is_available() else 'cuda'),
                project='my_test_eval_pic',
                name='test_evaluation'
            )
            print(f"評估完成！圖表位於 runs/detect/test_evaluation/")
            print(f"mAP@50: {metrics.box.map50:.4f}")
            print(f"mAP@50-95: {metrics.box.map:.4f}")
        except Exception as e:
            print(f"[錯誤] 評估過程發生錯誤 (可能是 YAML 路徑或資料問題): {e}")
    else:
        print(f"[嚴重警告] 建立 YAML 失敗，跳過圖表生成。")

    # 3. 產出畫框後的圖片 (Visualized Images)
    print("\n[Step 2] 正在對測試集畫框...")

    if os.path.exists(TEST_DIR_SOURCE):
        
        # 注意：若測試集很大，save=True 會佔用空間並花時間
        # 作業通常需要繳交部分結果或確認程式能跑
        with torch.no_grad():
            model.predict(
                source=TEST_DIR_SOURCE,
                save=True,
                save_txt=True,   # 若需要輸出座標檔
                imgsz=640,       # 配合 train.py 的設定
                conf=0.001,       # 推論時的信心度門檻
                iou=0.5,
                name='test_prediction',
                exist_ok=True,
                device = ('mps' if torch.backends.mps.is_available() else 'cuda'),
                project= 'my_results'
            )
        print(f"推論完成！畫框圖片位於 runs/detect/test_prediction/")
    else:
        print(f"[錯誤] 找不到測試圖片來源: {TEST_DIR_SOURCE}")

if __name__ == '__main__':
    run_test()
