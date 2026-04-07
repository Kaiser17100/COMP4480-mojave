from pathlib import Path
import cv2
from ultralytics import YOLO

# ---------------------------
# User-specific paths
# ---------------------------
MODEL_PATH = Path.home() / "Desktop" / "yolov2" / "runs" / "talon_pose_v1" / "weights" / "best.pt"

# Burayı kendi UDP/GStreamer pipeline'ına göre değiştir.
# Örnek pipeline:
# gst_pipeline = (
#     'udpsrc port=5600 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
#     'rtph264depay ! avdec_h264 ! videoconvert ! appsink drop=true sync=false'
# )
GST_PIPELINE = "YOUR_GSTREAMER_PIPELINE_HERE"

CONF = 0.25
IMG_SIZE = 640
SHOW = True


def compute_center_deviation(x1, y1, x2, y2, frame_w, frame_h):
    obj_cx = (x1 + x2) / 2.0
    obj_cy = (y1 + y2) / 2.0
    img_cx = frame_w / 2.0
    img_cy = frame_h / 2.0

    dx = obj_cx - img_cx   # + ise sağda, - ise solda
    dy = obj_cy - img_cy   # + ise aşağıda, - ise yukarıda

    dx_norm = dx / img_cx if img_cx != 0 else 0.0
    dy_norm = dy / img_cy if img_cy != 0 else 0.0

    return obj_cx, obj_cy, dx, dy, dx_norm, dy_norm


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    if GST_PIPELINE == "YOUR_GSTREAMER_PIPELINE_HERE":
        raise ValueError("Önce GST_PIPELINE değişkenine kendi GStreamer pipeline'ını yaz.")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("UDP/GStreamer stream açılamadı.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı.")
            break

        h, w = frame.shape[:2]
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        result = results[0]

        # Görüntü merkezi
        cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 255), -1)

        if result.boxes is not None and len(result.boxes) > 0:
            # En yüksek skorlu kutuyu al
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            best_idx = confs.argmax()

            x1, y1, x2, y2 = boxes_xyxy[best_idx]
            score = float(confs[best_idx])

            obj_cx, obj_cy, dx, dy, dx_norm, dy_norm = compute_center_deviation(
                x1, y1, x2, y2, w, h
            )

            # Bounding box çiz
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(obj_cx), int(obj_cy)), 5, (0, 0, 255), -1)
            cv2.line(frame, (w // 2, h // 2), (int(obj_cx), int(obj_cy)), (255, 0, 0), 2)

            text1 = f"conf={score:.2f}"
            text2 = f"dx={dx:.1f} px ({dx_norm:.3f}), dy={dy:.1f} px ({dy_norm:.3f})"
            cv2.putText(frame, text1, (int(x1), max(20, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Konsola yazdır
            print(
                f"Target center=({obj_cx:.1f}, {obj_cy:.1f}) | "
                f"dx={dx:.1f}px dy={dy:.1f}px | "
                f"dx_norm={dx_norm:.3f} dy_norm={dy_norm:.3f}"
            )

            # Pose keypoint'leri de istersen burada kullanabilirsin:
            # if result.keypoints is not None:
            #     kpts = result.keypoints.xy.cpu().numpy()[best_idx]
            #     for (kx, ky) in kpts:
            #         cv2.circle(frame, (int(kx), int(ky)), 4, (255, 255, 0), -1)

        if SHOW:
            cv2.imshow("YOLOv8 Pose UDP Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
