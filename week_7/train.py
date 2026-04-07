from ultralytics import YOLO
import os

# Yolun doğruluğunu kontrol edelim
yaml_path = r'/home/username/Desktop/dataset/data.yaml'
if not os.path.exists(yaml_path):
    print("HATA: data.yaml dosyası belirttiğin yolda bulunamadı!")
else:
    model = YOLO('yolo11n-pose.pt')
    model.train(data=yaml_path, epochs=10, imgsz=640, device=0)
