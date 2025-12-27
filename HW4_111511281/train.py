import os
import yaml
from ultralytics import YOLO
import shutil


STUDENT_ID = "111511281"
DATASET_ROOT = "./obj_data"  # 資料集根目錄

MODEL_CONFIG = "yolo11l.pt"
IMG_SIZE = 640
BATCH_SIZE =16  # A100 可以開 32 或 64
EPOCHS = 100
DEVICE = '0'

def create_simple_yaml():

    yaml_path = os.path.join(DATASET_ROOT, 'data_simple.yaml')

    

    yaml_content = {
        'path': os.path.abspath(DATASET_ROOT),      
        'train': 'train/images',
        'val': 'test/images',
        'test': 'test/images',
        'names': {0: 'pig'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"簡易版 YAML 已建立: {yaml_path}")
    print(f"Train路徑: {yaml_content['train']}")
    print(f"Val  路徑: {yaml_content['val']}")
    return yaml_path

if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'disabled'

    try:
        # 1. 產生最簡單的 YAML
        # split_data()
        yaml_path = create_simple_yaml()

        print(f"=== 開始訓練 (Model: {MODEL_CONFIG}) ===")
        model = YOLO(MODEL_CONFIG)

        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            
            # [原廠設定]
            pretrained=True,
            name=f"HW4_{STUDENT_ID}",
            workers=8,
            scale = 0.7,
            mosaic = 1,
            mixup = 0.6,
            copy_paste = 0.7,
            fliplr=0.5,     # 左右翻轉 (標準配備)
            flipud=0.5,
            hsv_h=0.04,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=15.0,
            save_period = 1,
            # 使用預設增強，不做任何修改
            plots=True
        )

        # 匯出
        best_weight = f"runs/detect/HW4_{STUDENT_ID}/weights/best.pt"
        if os.path.exists(best_weight):
            target = f"{STUDENT_ID}.pt"
            shutil.copy(best_weight, target)
            print(f"訓練完成！模型已匯出至 {target}")


    except Exception as e:
        print(f"Error: {e}")