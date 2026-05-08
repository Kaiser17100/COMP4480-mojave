import time, socket, subprocess, sys, os
import webview

# Sunucunun gösterdiği panel adresi
HOST_IP = socket.gethostbyname(socket.gethostname())  # kendi IP adresini otomatik bulur
URL = f"http://{HOST_IP}:10001/dashboard"   

def wait_port(host=HOST_IP, port=10001, timeout=20):
    start = time.time()
    while time.time()-start < timeout:
        try:
            # Sunucu ayağa kalktı mı?
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fake_server_path = os.path.join(script_dir, "fake_server.py")
    srv = subprocess.Popen([sys.executable, fake_server_path], cwd=script_dir)
    try:
        wait_port()
        class Api:
            def open_video_wall(self):
                for w in webview.windows:
                    if w.title == "Video Wall":
                        return
                webview.create_window(
                    "Video Wall",
                    f"http://{HOST_IP}:10001/video_wall_static",
                    width=1200,
                    height=800,
                    on_top=True
                )
            def open_history(self):
                for w in webview.windows:
                    if w.title == "Geçmiş Kayıtlar":
                        return
                webview.create_window(
                    "Geçmiş Kayıtlar",
                    f"http://{HOST_IP}:10001/history",
                    width=1200,
                    height=800
                )
        main_window = webview.create_window(
            "IHA Telemetri Dashboard",
            URL,
            width=1200,
            height=800,
            js_api=Api()
        )
        webview.start()
    finally:
        srv.terminate()