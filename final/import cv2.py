import cv2
import numpy as np
import socket
import subprocess
import threading
import time
import os

# =========================
# RTP/H264 UDP GİRİŞ
# =========================
UDP_IN_PORT = int(os.getenv("IHA_CAMERA_IN_PORT", "5604"))

# =========================
# ÇIKIŞ JPEG UDP
# =========================
TEAM_NO = int(os.getenv("IHA_TEAM_NO", "4"))
STREAM_PORTS = {
    20: 5420,
    1: 5425,
    2: 5426,
    3: 5427,
    4: 5428,
    5: 5429,
}
UDP_OUT_IP = os.getenv("IHA_VIDEO_HOST", "127.0.0.1")
UDP_OUT_PORT = int(os.getenv("IHA_VIDEO_PORT", str(STREAM_PORTS.get(TEAM_NO, 5400 + TEAM_NO))))

# =========================
# Görüntü boyutu
# =========================
WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 50

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

click_point = None


def on_mouse_click(event, x, y, flags, param):
    global click_point

    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"Hedef Tıklandı: {x}, {y}")


def process_frame(frame):
    global click_point

    h, w, _ = frame.shape

    # Merkez çizmek istersen aç
    # cv2.circle(frame, (w // 2, h // 2), 5, (0, 255, 0), -1)

    # Tıklanan noktayı çizmek istersen aç
    # if click_point is not None:
    #     cv2.circle(frame, click_point, 10, (0, 0, 255), 2)
    #     cv2.line(frame, (w // 2, h // 2), click_point, (255, 0, 0), 1)

    return frame


def create_sdp_file():
    sdp_text = f"""v=0
o=- 0 0 IN IP4 127.0.0.1
s=RTP H264 Stream
c=IN IP4 0.0.0.0
t=0 0
m=video {UDP_IN_PORT} RTP/AVP 96
a=rtpmap:96 H264/90000
a=recvonly
"""

    sdp_path = "/tmp/udp_h264_stream.sdp"

    with open(sdp_path, "w") as f:
        f.write(sdp_text)

    print("SDP dosyası oluşturuldu:")
    print(sdp_path)
    print(sdp_text)

    return sdp_path


def print_ffmpeg_errors(process):
    """
    FFmpeg stderr çıktısını ayrı thread ile okur.
    Böylece hata sebebini terminalde görürüz.
    """
    while True:
        line = process.stderr.readline()

        if not line:
            break

        try:
            print("[FFmpeg]", line.decode(errors="ignore").strip())
        except Exception:
            pass


def start_ffmpeg():
    sdp_path = create_sdp_file()

    command = [
        "ffmpeg",

        "-hide_banner",
        "-loglevel", "warning",

        "-protocol_whitelist", "file,udp,rtp",

        # gecikmeyi azaltmak için
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "1000000",
        "-probesize", "1000000",

        # SDP üzerinden RTP/H264 oku
        "-i", sdp_path,

        # OpenCV için sabit boyut ve BGR raw çıkış
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-"
    ]

    print("FFmpeg başlatılıyor:")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )

    stderr_thread = threading.Thread(
        target=print_ffmpeg_errors,
        args=(process,),
        daemon=True
    )
    stderr_thread.start()

    return process


def main():
    print(f"FFmpeg RTP/H264 görüntü dinliyor: 0.0.0.0:{UDP_IN_PORT}")
    print(f"Python JPEG olarak tekrar yayınlıyor: {UDP_OUT_IP}:{UDP_OUT_PORT}")

    ffmpeg_process = start_ffmpeg()

    frame_size = WIDTH * HEIGHT * 3

    window_name = "Mini Talon Vision - FFmpeg RTP/H264 Input"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    last_error_time = 0

    try:
        while True:
            if ffmpeg_process.poll() is not None:
                print("FFmpeg kapandı. Stream açılamadı veya giriş formatı uyumsuz.")
                break

            raw_frame = ffmpeg_process.stdout.read(frame_size)

            if len(raw_frame) != frame_size:
                now = time.time()

                if now - last_error_time > 1.0:
                    print("Görüntü frame'i eksik geldi veya FFmpeg henüz frame üretmedi.")
                    print(f"Beklenen byte: {frame_size}, gelen byte: {len(raw_frame)}")
                    last_error_time = now

                time.sleep(0.01)
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((HEIGHT, WIDTH, 3))

            frame = process_frame(frame)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                print("Çıkış yapılıyor...")
                break

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if success:
                send_sock.sendto(buffer.tobytes(), (UDP_OUT_IP, UDP_OUT_PORT))

    except KeyboardInterrupt:
        print("Durduruluyor...")

    finally:
        send_sock.close()

        if ffmpeg_process is not None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
