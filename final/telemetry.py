import requests
import time
from datetime import datetime, timedelta

def now_clock():
    now = datetime.utcnow()
    return {
        "saat": now.hour,
        "dakika": now.minute,
        "saniye": now.second,
        "milisaniye": int(now.microsecond / 1000)
    }

def login(session: requests.Session, base_url: str, username: str, password: str) -> str:
    url = f"{base_url}/api/giris"
    payload = {
        "kadi": username,
        "sifre": password
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


def get_sunucu_saati(session: requests.Session, base_url: str):
    url = f"{base_url}/api/sunucusaati"
    r = session.get(url, timeout=5)
    r.raise_for_status()
    return r.json().get("sunucusaati")


def get_hss(session: requests.Session, base_url: str, token: str):
    url = f"{base_url}/api/hss_koordinatlari"
    headers = {"Authorization": f"Bearer {token}"}
    r = session.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json().get("hss_koordinat_bilgileri")


def get_qr(session: requests.Session, base_url: str, token: str):
    print("SEND QR")
    url = f"{base_url}/api/qr_koordinati"
    headers = {"Authorization": f"Bearer {token}"}
    r = session.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()


def send_telemetry(
    session: requests.Session,
    base_url: str,
    token: str,
    team_no: int,
    iha_enlem: float,
    iha_boylam: float,
    iha_irtifa: float,
    iha_dikilme: float,
    iha_yonelme: float,
    iha_yatis: float,
    iha_hiz: float,
    iha_batarya: float,
    iha_otonom: int,
    gps_saati: dict,
    iha_kilitlenme: int = 0,
    hedef_merkez_X: int = 0,
    hedef_merkez_Y: int = 0,
    hedef_genislik: int = 0,
    hedef_yukseklik: int = 0
):
    url = f"{base_url}/api/telemetri_gonder"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "takim_numarasi": team_no,
        "iha_enlem": iha_enlem,
        "iha_boylam": iha_boylam,
        "iha_irtifa": iha_irtifa,
        "iha_dikilme": iha_dikilme,
        "iha_yonelme": iha_yonelme,
        "iha_yatis": iha_yatis,
        "iha_hiz": iha_hiz,
        "iha_batarya": iha_batarya,
        "iha_otonom": iha_otonom,
        "iha_kilitlenme": iha_kilitlenme,
        "hedef_merkez_X": hedef_merkez_X,
        "hedef_merkez_Y": hedef_merkez_Y,
        "hedef_genislik": hedef_genislik,
        "hedef_yukseklik": hedef_yukseklik,
        "gps_saati": gps_saati
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print("TELEMETRY STATUS:", r.status_code)
    return r


def send_lock(
    session: requests.Session, 
    base_url: str, 
    token: str, 
    otonom_kilitlenme: int,
    kitlenmeBitis: dict
):
    url = f"{base_url}/api/kilitlenme_bilgisi"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "kilitlenmeBitisZamani": kitlenmeBitis,
        "otonom_kilitlenme": otonom_kilitlenme,
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print("LOCK STATUS:", r.status_code, "| RESP:", r.text)


def send_kamikaze(
    session: requests.Session, 
    base_url: str, 
    token: str, 
    team_no: int, 
    qr_text: str, 
    kamikazeBaslangicZamani: dict,
    kamikazeBitisZamani: dict
):
    url = f"{base_url}/api/kamikaze_bilgisi"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "kaynak_takim": team_no,
        "qrMetni": qr_text,
        "kamikazeBaslangicZamani": kamikazeBaslangicZamani,
        "kamikazeBitisZamani": kamikazeBitisZamani
    }

    r = session.post(url, headers=headers, json=payload, timeout=5)
    print("KAMIKAZE STATUS:", r.status_code, "| RESP:", r.text)
