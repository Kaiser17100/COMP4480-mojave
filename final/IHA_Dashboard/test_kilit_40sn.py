import requests
import time
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:10001"

USERNAME = "3"
PASSWORD = "3"

TEAM_NO = 3

TOTAL_DURATION = 40.0
TELEMETRY_PERIOD = 0.5   # 2 Hz
LOCK_PERIOD = 4.0        # her 4 sn
KAMIKAZE_PERIOD = 6.0    # her 6 sn

session = requests.Session()


def now_clock():
    now = datetime.utcnow()
    return {
        "saat": now.hour,
        "dakika": now.minute,
        "saniye": now.second,
        "milisaniye": int(now.microsecond / 1000)
    }


def shifted_clock(ms_after=0):
    future = datetime.utcnow() + timedelta(milliseconds=ms_after)
    return {
        "saat": future.hour,
        "dakika": future.minute,
        "saniye": future.second,
        "milisaniye": int(future.microsecond / 1000)
    }


def login():
    url = f"{BASE_URL}/api/giris"
    payload = {
        "kadi": USERNAME,
        "sifre": PASSWORD
    }

    r = session.post(url, json=payload, timeout=5)
    print("LOGIN STATUS:", r.status_code)
    print("LOGIN RESP:", r.text)
    r.raise_for_status()

    data = r.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("Login başarılı ama token gelmedi.")

    return token


def send_telemetry(token, seq):
    url = f"{BASE_URL}/api/telemetri_gonder"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    lat = 38.700770 + (seq * 0.00001)
    lon = 27.454110 + (seq * 0.00001)

    payload = {
        "takim_numarasi": TEAM_NO,
        "iha_enlem": lat,
        "iha_boylam": lon,
        "iha_irtifa": 120,
        "iha_dikilme": 2,
        "iha_yonelme": 90,
        "iha_yatis": 1,
        "iha_hiz": 18,
        "iha_batarya": 75,
        "iha_otonom": 1,
        "iha_kilitlenme": 1,
        "hedef_merkez_X": 320,
        "hedef_merkez_Y": 240,
        "hedef_genislik": 120,
        "hedef_yukseklik": 90,
        "gps_saati": now_clock()
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print(f"TELEMETRY [{seq}] STATUS:", r.status_code, "| RESP:", r.text)


def send_lock(token, lock_no):
    url = f"{BASE_URL}/api/kilitlenme_bilgisi"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "kilitlenmeBitisZamani": now_clock(),
        "otonom_kilitlenme": 1,
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print(f"LOCK [{lock_no}] STATUS:", r.status_code, "| RESP:", r.text)


def send_kamikaze(token, kz_no):
    url = f"{BASE_URL}/api/kamikaze_bilgisi"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "kaynak_takim": TEAM_NO,
        "qrMetni": f"TEST-QR-{kz_no}",
        "kamikazeBaslangicZamani": now_clock(),
        "kamikazeBitisZamani": shifted_clock(800)
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print(f"KAMIKAZE [{kz_no}] STATUS:", r.status_code, "| RESP:", r.text)


def main():
    token = login()
    print("TOKEN:", token)

    start_time = time.time()
    next_telemetry = start_time
    next_lock = start_time + LOCK_PERIOD
    next_kamikaze = start_time + KAMIKAZE_PERIOD

    seq = 0
    lock_no = 0
    kz_no = 0

    while time.time() - start_time < TOTAL_DURATION:
        now = time.time()

        if now >= next_telemetry:
            seq += 1
            try:
                send_telemetry(token, seq)
            except Exception as e:
                print("TELEMETRY ERROR:", e)
            next_telemetry += TELEMETRY_PERIOD

        if now >= next_lock:
            lock_no += 1
            try:
                send_lock(token, lock_no)
            except Exception as e:
                print("LOCK ERROR:", e)
            next_lock += LOCK_PERIOD

        if now >= next_kamikaze:
            kz_no += 1
            try:
                send_kamikaze(token, kz_no)
            except Exception as e:
                print("KAMIKAZE ERROR:", e)
            next_kamikaze += KAMIKAZE_PERIOD

        time.sleep(0.05)

    print("BİTTİ: 40 saniyelik test tamamlandı.")


if __name__ == "__main__":
    main()