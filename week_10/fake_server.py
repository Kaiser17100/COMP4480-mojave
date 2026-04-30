import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify, make_response, send_file, Response, send_from_directory
from datetime import datetime, timezone
import time, math, random
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import secrets
import json
import hashlib
import subprocess
import signal
import threading

app = Flask(__name__)

# =============================================
#  STREAM MANAGER  –  Native ffplay pencereleri
# =============================================
# Her takım için ayrı bir ffplay penceresi açılır.
# Doğrudan UDP → decode → ekran. Sıfır ara katman, sıfır kasma.

STREAM_PROCS = {}        # takim_id -> { "proc": Popen, "ip": str, "port": int }
STREAM_LOCK  = threading.Lock()

LAST_VIDEO_TS = {}   # team -> son görüntü zamanı
VIDEO_OK = {}        # team -> son probe'da görüntü var mı

SCORING_ACTIVE = False   # başlangıçta kapalı

# Varsayılan takım port eşlemeleri (UI'dan geçersiz kılınabilir)
DEFAULT_STREAM_PORTS = {
    20: 5420,
    25: 5425,
    26: 5426,
    27: 5427,
    28: 5428,
    29: 5429,
}
DEFAULT_STREAM_IP = "0.0.0.0"


def _start_ffplay(takim_id, ip, port):
    """ffplay penceresini başlatır. Doğrudan UDP → ekran."""
    window_title = f"Takim {takim_id} - udp://{ip}:{port}"

    cmd = [
        "ffplay",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-window_title", window_title,
        "-x", "640",
        "-y", "480",
        "-i", f"udp://{ip}:{port}"
    ]

    print(f"🎬 ffplay komutu: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    import time as _time
    _time.sleep(0.3)
    if proc.poll() is not None:
        print(f"❌ ffplay anında kapandı! Exit code: {proc.returncode}")
    else:
        print(f"✅ ffplay çalışıyor PID={proc.pid}  pencere: {window_title}")

    return proc


def _stop_ffplay(takim_id):
    """Belirtilen takımın ffplay sürecini durdurur."""
    with STREAM_LOCK:
        info = STREAM_PROCS.pop(takim_id, None)
    if info and info["proc"]:
        try:
            info["proc"].terminate()
            info["proc"].wait(timeout=3)
        except Exception:
            try:
                info["proc"].kill()
            except Exception:
                pass



@app.route("/api/stream/start/<int:takim_id>", methods=["POST"])
def api_stream_start(takim_id):
    """Bir takımın ffplay penceresini açar."""
    data = request.get_json(silent=True) or {}
    ip   = data.get("ip", DEFAULT_STREAM_IP)
    port = data.get("port", DEFAULT_STREAM_PORTS.get(takim_id, 5400 + takim_id))

    # Zaten çalışıyorsa önce durdur
    _stop_ffplay(takim_id)

    proc = _start_ffplay(takim_id, ip, port)
    with STREAM_LOCK:
        STREAM_PROCS[takim_id] = {"proc": proc, "ip": ip, "port": port}

    print(f"▶️  Stream BAŞLATILDI: Takım {takim_id}  udp://{ip}:{port}")
    return jsonify({"ok": True, "takim": takim_id, "ip": ip, "port": port}), 200


@app.route("/api/stream/stop/<int:takim_id>", methods=["POST"])
def api_stream_stop(takim_id):
    """Bir takımın ffplay penceresini kapatır."""
    _stop_ffplay(takim_id)
    print(f"⏹️  Stream DURDURULDU: Takım {takim_id}")
    return jsonify({"ok": True, "takim": takim_id}), 200


@app.route("/api/stream/status", methods=["GET"])
def api_stream_status():
    """Aktif stream'lerin durumunu döndürür."""
    with STREAM_LOCK:
        active = {}
        for tid, info in list(STREAM_PROCS.items()):
            alive = info["proc"].poll() is None
            active[str(tid)] = {
                "ip": info["ip"],
                "port": info["port"],
                "alive": alive,
                "pid": info["proc"].pid
            }
    return jsonify({"ok": True, "streams": active}), 200


def _udp_probe(ip, port, timeout=1.5):
    """
    Belirtilen UDP portunda kısa süre dinler, veri gelip gelmediğine bakar.
    Native thread'de çalıştırılmalı (eventlet uyumluluğu için).
    """
    import socket as _socket
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.bind((ip, port))
        data, addr = sock.recvfrom(4096)
        return {"has_data": True, "bytes": len(data), "source": f"{addr[0]}:{addr[1]}"}
    except _socket.timeout:
        return {"has_data": False, "bytes": 0, "source": None}
    except OSError as e:
        return {"has_data": False, "bytes": 0, "source": None, "error": str(e)}
    finally:
        sock.close()


@app.route("/api/stream/probe", methods=["POST"])
def api_stream_probe():
    """
    Belirtilen UDP portunda veri olup olmadığını kontrol eder.
    Body: { "ip": "0.0.0.0", "port": 5425 }
    """
    data = request.get_json(silent=True) or {}
    ip   = data.get("ip", "0.0.0.0")
    port = data.get("port")
    if not port:
        return jsonify({"ok": False, "error": "port gerekli"}), 400

    try:
        result = _udp_probe(ip, int(port))
    except Exception as e:
        result = {"has_data": False, "bytes": 0, "error": str(e)}

    # Takımı bul
    team = None
    try:
        # Önce body'den geldiyse onu kullan
        if data.get("takim_id") is not None:
            team = int(data.get("takim_id"))
        else:
            # Porttan takım eşlemesi bul
            for tid, p in DEFAULT_STREAM_PORTS.items():
                if int(p) == int(port):
                    team = int(tid)
                    break
    except Exception as e:
        print("❌ team bulma hatası:", e)

    # Görüntü durumu güncelle
    if team is not None:
        if result.get("has_data"):
            LAST_VIDEO_TS[team] = time.time()
            VIDEO_OK[team] = True

            try:
                _ensure_team(team)

                # sadece ilk kez 1 yap, sürekli artmasın
                if SCORING_ACTIVE and SCORES[team]["video_tx"] == 0:
                    SCORES[team]["video_tx"] = 1
                    _recalc_total(team)
                    emit_score_update()
                    print(f"🎯 Görüntü algılandı → Takım {team} video_tx=1, +50 puan")
            except Exception as e:
                print("❌ video_tx puanlama hatası:", e)
        else:
            VIDEO_OK[team] = False

    return jsonify({"ok": True, "port": port, **result}), 200

CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")  # dev için *; prod’da domain kısıtla

# Basit kimlik & oturum
VALID_USERS = [
    {"kadi": "anafarta", "sifre": "123", "takim": 25},
    {"kadi": "yem", "sifre": "123456", "takim": 20},
    {"kadi": "deneme", "sifre": "deneme", "takim": 26},
    {"kadi": "Mojave", "sifre": "123", "takim": 4},
    {"kadi": "1", "sifre": "1", "takim": 1},
    {"kadi": "2", "sifre": "2", "takim": 2},
    {"kadi": "3", "sifre": "3", "takim": 3},
    {"kadi": "4", "sifre": "4", "takim": 4},
    {"kadi": "5", "sifre": "5", "takim": 5}
]

def video_probe_loop():
    """
    Puanlama açıkken yayınları periyodik kontrol eder.
    Yayın ilk kez algılanırsa takım başına sadece 1 kez görüntü puanı verir.
    """
    while True:
        try:
            # Puanlama kapalıysa sadece bekle
            if not SCORING_ACTIVE:
                socketio.sleep(2.0)
                continue

            changed = False

            for team, port in DEFAULT_STREAM_PORTS.items():
                try:
                    _ensure_team(team)

                    # Zaten görüntü puanı aldıysa tekrar bakmaya gerek yok
                    if SCORES[team]["video_tx"] != 0:
                        continue

                    result = _udp_probe(DEFAULT_STREAM_IP, int(port), timeout=0.5)

                    if result.get("has_data"):
                        LAST_VIDEO_TS[team] = time.time()
                        VIDEO_OK[team] = True

                        SCORES[team]["video_tx"] = 1
                        _recalc_total(team)
                        changed = True

                        print(f"🎯 Yayın algılandı -> Takım {team} video_tx=1, +50 puan")
                    else:
                        VIDEO_OK[team] = False

                except Exception as e:
                    print(f"❌ video_probe_loop takım {team} hata: {e}")

            if changed:
                emit_score_update()

        except Exception as e:
            print("❌ video_probe_loop genel hata:", e)

        socketio.sleep(2.0)

ISSUED_TOKENS = {}  # token → team

# In-memory en son telemetri kayıtları: {takim_numarasi: {"telemetry": t, "ts": time.time()}}
_latest_telemetry = {}

# (Opsiyonel) kaç saniyeden eski telemetry'i düşman listesinden çıkarmak istersin
_TELEMETRY_STALE_SEC = 5.0  # örn. 5 saniye; gerçek testte network koşullarına göre arttırabilirsin
TEAM_NO = None

# HSS'ler uçağa gönderilsin mi?
HSS_SEND_ENABLED = True

# HSS sisteminin aktif/pasif durumu (UI'den kontrol edilebilir)
HSS_SYSTEM_ACTIVE = False  # Başlangıçta KAPALI

#TOKEN = "fake_token_123"
SESSION_COOKIE = "sessionid"

# 2 Hz limiti (takım bazlı)
_last_telemetry_ts = {}
_RATE_PERIOD = 0.5  # saniye → 2 Hz

MIN_TELEMETRY_PERIOD = 1.0   # takım en geç 1 sn içinde veri göndermeli
TEAM_1HZ_VIOLATION = {}      # team -> bool

LAST_ACCEPTED_TELEMETRY_REAL_TS = {}

# --- HSS pencere kontrolü ---
_HSS_EMPTY1_SEC = 10  # ilk 10 saniye boş
_HSS_ACTIVE_SEC = 10  # sonraki 10 saniye dolu
_SERVER_START_MONO = time.monotonic()




# --- SQLite setup ---
import os, sqlite3, json

DB_PATH = os.path.join(os.path.dirname(__file__), "iha_logs.db")


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc TEXT NOT NULL,
        takim INTEGER NOT NULL,
        enlem REAL,
        boylam REAL,
        irtifa REAL,
        hiz REAL,
        batarya INTEGER,
        raw_json TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_takim_ts ON telemetry(takim, ts_utc);")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS locks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    takim INTEGER NOT NULL,
                    saat INTEGER NOT NULL,
                    dakika INTEGER NOT NULL,
                    saniye INTEGER NOT NULL,
                    milisaniye INTEGER NOT NULL,
                    otonom_kilitlenme INTEGER NOT NULL,
                    ts_utc TEXT NOT NULL
                );
                """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_locks_ts ON locks(ts_utc);")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS kamikaze (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,              -- ISO8601 UTC (server zamanı)
            kaynak_takim INTEGER,              -- gönderende varsa
            qr_metni TEXT NOT NULL,
            baslangic_gps TEXT NOT NULL,       -- JSON string
            bitis_gps TEXT NOT NULL,           -- JSON string
            extra_json TEXT                    -- ham paket / ek alanlar
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kamikaze_ts ON kamikaze(ts_utc)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS fences (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      kind TEXT NOT NULL,            -- 'polygon' (dikdörtgeni GeoJSON Polygon tutacağız)
      geojson TEXT NOT NULL,         -- GeoJSON Feature
      color TEXT DEFAULT '#ef4444',  -- kırmızı
      updated_at TEXT NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fences_updated ON fences(updated_at);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS hss (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      lat REAL NOT NULL,
      lon REAL NOT NULL,
      radius REAL NOT NULL,     -- metre
      active INTEGER NOT NULL DEFAULT 1,
      updated_at TEXT NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hss_active ON hss(active);")

    con.commit()
    con.close()
    print("✅ DB hazır:", DB_PATH)


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def list_fences():
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    rows = cur.execute("SELECT id,name,kind,geojson,color,updated_at FROM fences ORDER BY id").fetchall()
    con.close()
    keys = ["id", "name", "kind", "geojson", "color", "updated_at"]
    out = [dict(zip(keys, r)) for r in rows]
    for r in out:
        try:
            r["geojson"] = json.loads(r["geojson"])
        except:
            pass
    return out


def list_hss():
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    rows = cur.execute("SELECT id,name,lat,lon,radius,active,updated_at FROM hss WHERE active=1 ORDER BY id").fetchall()
    con.close()
    keys = ["id", "name", "lat", "lon", "radius", "active", "updated_at"]
    return [dict(zip(keys, r)) for r in rows]


def distance_m(lat1, lon1, lat2, lon2):
    """Iki enlem/boylam arasındaki mesafeyi metre cinsinden hesapla (haversine)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_inside_hss(lat, lon):
    """Aktif HSS'lerden herhangi birinin içinde mi?"""
    try:
        for it in list_hss():
            h_lat = float(it["lat"])
            h_lon = float(it["lon"])
            r = float(it["radius"])
            d = distance_m(lat, lon, h_lat, h_lon)
            if d <= r:
                return True
    except Exception as e:
        print("⚠️ is_inside_hss hata:", e)
    return False


def insert_hss(name, lat, lon, radius):
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("INSERT INTO hss(name,lat,lon,radius,active,updated_at) VALUES (?,?,?,?,1,?)",
                (name, float(lat), float(lon), float(radius), now_iso()))
    con.commit();
    _id = cur.lastrowid;
    con.close()
    return _id


def update_hss(hid, **fields):
    if not fields: return
    sets, args = [], []
    for k in ("name", "lat", "lon", "radius", "active"):
        if k in fields: sets.append(f"{k}=?"); args.append(fields[k])
    sets.append("updated_at=?");
    args.append(now_iso())
    args.append(int(hid))
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute(f"UPDATE hss SET {', '.join(sets)} WHERE id=?", args)
    con.commit();
    con.close()


def delete_hss(hid):
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("DELETE FROM hss WHERE id=?", (int(hid),))
    con.commit();
    con.close()


def save_kamikaze_row(payload: dict):
    """/api/kamikaze_bilgisi gelen paketi kalıcı kaydet"""
    import json, datetime
    ts_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    kaynak = payload.get("kaynak_takim")  # yoksa None kalır
    qr = payload.get("qrMetni")
    kb = payload.get("kamikazeBaslangicZamani", {})
    ke = payload.get("kamikazeBitisZamani", {})

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO kamikaze (ts_utc, kaynak_takim, qr_metni, baslangic_gps, bitis_gps, extra_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ts_utc, kaynak, qr, json.dumps(kb), json.dumps(ke), json.dumps(payload)))
    conn.commit()
    conn.close()


def query_kamikaze_history(kaynak=None, start=None, end=None, limit=1000):
    """UI tarih filtresi için socket’ten sorgulanır"""
    import json
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = "SELECT ts_utc, kaynak_takim, qr_metni, baslangic_gps, bitis_gps FROM kamikaze WHERE 1=1"
    args = []
    if kaynak not in (None, "", "null"):
        sql += " AND kaynak_takim = ?"
        args.append(int(kaynak))
    if start:
        sql += " AND ts_utc >= ?";
        args.append(start)
    if end:
        sql += " AND ts_utc <= ?";
        args.append(end)
    sql += " ORDER BY ts_utc DESC LIMIT ?"
    args.append(int(limit))

    rows = [dict(r) for r in cur.execute(sql, args).fetchall()]
    # JSON alanları dict’e çevir
    for r in rows:
        try:
            r["baslangic_gps"] = json.loads(r["baslangic_gps"]) if r["baslangic_gps"] else None
        except:
            pass
        try:
            r["bitis_gps"] = json.loads(r["bitis_gps"]) if r["bitis_gps"] else None
        except:
            pass
    conn.close()
    return rows


def query_telemetry_history(takim=None, start_iso=None, end_iso=None, limit=1000):
    """
    telemetry tablosundan filtreli veri çeker.
    start_iso / end_iso -> 'YYYY-MM-DDTHH:MM:SSZ' gibi ISO (UTC)
    """
    q = "SELECT ts_utc, takim, enlem, boylam, irtifa, hiz, batarya FROM telemetry WHERE 1=1"
    params = []
    if takim is not None and str(takim).strip() != "":
        q += " AND takim = ?"
        params.append(int(takim))
    if start_iso:
        q += " AND ts_utc >= ?"
        params.append(start_iso)
    if end_iso:
        q += " AND ts_utc <= ?"
        params.append(end_iso)
    q += " ORDER BY ts_utc DESC"
    if limit:
        q += f" LIMIT {int(limit)}"

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute(q, params).fetchall()
    con.close()

    # dict listeye çevir
    keys = ["ts_utc", "takim", "enlem", "boylam", "irtifa", "hiz", "batarya"]
    return [dict(zip(keys, r)) for r in rows]


def query_locks_history(kaynak=None, start_iso=None, end_iso=None, limit=1000):
    import sqlite3

    q = """
        SELECT ts_utc,
               kaynak_takim,
               kilitlenen_takim,
               otonom_kilitlenme,
               kilit_bitis_gps
        FROM locks
        WHERE 1=1
    """
    params = []

    if kaynak not in (None, "", "null", "None"):
        q += " AND kaynak_takim = ?"
        params.append(int(kaynak))

    if start_iso:
        q += " AND ts_utc >= ?"
        params.append(start_iso)

    if end_iso:
        q += " AND ts_utc <= ?"
        params.append(end_iso)

    q += " ORDER BY ts_utc DESC"

    if limit:
        q += f" LIMIT {int(limit)}"

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute(q, params).fetchall()
    con.close()

    keys = [
        "ts_utc",
        "kaynak_takim",
        "kilitlenen_takim",
        "otonom_kilitlenme",
        "kilit_bitis_gps"
    ]

    return [dict(zip(keys, r)) for r in rows]



def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_telemetry_row(takim, t):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO telemetry (ts_utc, takim, enlem, boylam, irtifa, hiz, batarya, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        utc_now(), int(takim),
        float(t.get("iha_enlem", 0.0)),
        float(t.get("iha_boylam", 0.0)),
        float(t.get("iha_irtifa", 0.0)),
        float(t.get("iha_hiz", 0.0)),
        int(t.get("iha_batarya", 0)),
        json.dumps(t, ensure_ascii=False)
    ))
    con.commit()
    con.close()
    print("📝 telemetry→DB takım=", takim)


def save_lock_row(team: int, payload: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    kb = payload.get("kilitlenmeBitisZamani", {})

    kilit_bitis_json = json.dumps(kb)

    cur.execute("""
        INSERT INTO locks (
            ts_utc,
            kaynak_takim,
            kilitlenen_takim,
            otonom_kilitlenme,
            kilit_bitis_gps,
            extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        utc_now(),
        team,                                      # kaynak_takim
        payload.get("kilitlenen_takim", 0),        # varsa
        int(payload.get("otonom_kilitlenme", 0)),
        kilit_bitis_json,
        json.dumps(payload)                        # tüm payload saklanır
    ))

    con.commit()
    con.close()


@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route("/dashboard")
def dashboard():
    return send_file("ui.html")


def server_now_dict():
    now = datetime.now(timezone.utc)
    return {"gun": now.day, "saat": now.hour, "dakika": now.minute,
            "saniye": now.second, "milisaniye": int(now.microsecond / 1000)}

def ok_auth():
    hdr = request.headers.get("Authorization", "")
    if not hdr.startswith("Bearer "):
        return False

    tok = hdr.split(" ", 1)[1].strip()

    # TEST MODE: fake token da kabul
    if tok == "fake_token_123":
        return True

    # Gerçek login tokenı
    if tok in ISSUED_TOKENS:
        return True

    return False

# =========================
#  PUANLAMA (TEST - limitsiz)
# =========================

# team -> skor dökümü
SCORES = {}  # {team: {"auto_lock":0,"manual_lock":0,"kamikaze":0,"hss_sec":0,"tel_err_sec":0,"ihrac":0,"total":0.0}}
OFFLINE_AFTER_SEC = 2.5   # 2.5 sn telemetri yoksa offline say (rate-limit 0.5s olduğu için makul)


# Anlık durumlar
HSS_INSIDE = {}          # team -> bool
TEL_LAST_SEEN = {}       # team -> time.time()
TEL_LAST_HASH = {}       # team -> str
TEL_LAST_CHANGE = {}     # team -> time.time()
TEL_WARNED_AT = {}       # team -> time.time()   (ikaz edildi -> 10 sn say)
IHRAC_DONE = set()       # bir kez ihraç yaz (tekrar tekrar ceza yazmasın)


HSS_WARNED = set()   # HSS içinde olan takımlar (uyarı için)
GEOFENCE_OUTSIDE = {}      # team -> bool
GEOFENCE_WARNED_AT = {}    # team -> ts
GEOFENCE_IHRAC_DONE = set()
GEOFENCE_LAST = {}         # team -> last geofence check ts (watchdog için opsiyonel)

TEL_LAND_CALL = set()
TEL_LAND_REASONS = {}

TEAM_STATE = {}

TEL_STATE = {}
# team -> {
#   "mode": "ok" | "sabit" | "kesik",
#   "start": timestamp,
#   "ihrac_written": False
# }
FLIGHT_STATE = {}


def telemetry_signature(t):
    def r(x):
        try:
            return round(float(x), 6)
        except:
            return x

    sig_obj = {
        "lat": r(t.get("iha_enlem")),
        "lon": r(t.get("iha_boylam")),
        "alt": r(t.get("iha_irtifa")),
        "yaw": r(t.get("iha_yonelme")),
        "pitch": r(t.get("iha_dikilme")),
        "roll": r(t.get("iha_yatis")),
        "spd": r(t.get("iha_hiz")),
    }

    return hashlib.md5(
        json.dumps(sig_obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


def land_call(team, reason):
    if team not in TEL_LAND_CALL:
        TEL_LAND_CALL.add(team)
        TEL_LAND_REASONS[team] = reason

        socketio.emit("land_call_event", {
            "team": team,
            "reason": reason,
            "time": server_now_dict()
        })


def _ensure_team(team: int):
    if team is None:
        return
    if team not in SCORES:
        SCORES[team] = {
            # POZİTİF
            "auto_lock": 0,  # * 500
            "manual_lock": 0,  # * 50
            "kamikaze": 0,  # * 300
            "auto_land": 0,  # * 100
            "auto_takeoff": 0,  # * 50
            "video_tx": 0,  # * 50

            # CEZA
            "tel_err_sec": 0,  # * -0.2
            "hatali_kilit_kamikaze": 0,  # -30
            "manual_limit": 0,  # -10
            "hss_sec": 0,  # * -5

            # TEK SEFERLİK CEZA
            "area_violation": 0,  # * -150
            "boundary_violation": 0,  # * -200
            "ihrac": 0,  # * -75
            "total": 0.0
        }
    if team not in TEL_STATE:
        TEL_STATE[team] = {
            "mode": "ok",
            "start": time.time(),
            "ihrac_written": False
        }


def auth_team():
    """Bearer token -> takım numarası (fake_token_123 ise None döner)"""
    hdr = request.headers.get("Authorization", "")
    if not hdr.startswith("Bearer "):
        return None
    tok = hdr.split(" ", 1)[1].strip()
    # fake token testte var ama takım bilgisi yok
    if tok == "fake_token_123":
        return None
    return ISSUED_TOKENS.get(tok)

def emit_score_update():
    rows = []

    for team in sorted(SCORES.keys()):
        s = SCORES[team]
        rows.append({
            "team": team,

            "auto_lock": s["auto_lock"],
            "manual_lock": s["manual_lock"],
            "kamikaze": s["kamikaze"],
            "auto_land": s["auto_land"],
            "auto_takeoff": s["auto_takeoff"],
            "video_tx": s["video_tx"],

            "tel_err_sec": s["tel_err_sec"],
            "hatali_kilit_kamikaze": s["hatali_kilit_kamikaze"],  # 🔴 KRİTİK
            "manual_limit": s["manual_limit"],
            "hss_sec": s["hss_sec"],

            "area_violation": s["area_violation"],
            "boundary_violation": s["boundary_violation"],
            "ihrac": s["ihrac"],

            "total": s["total"]
        })

    socketio.emit("score_update", {
        "scores": rows,
        "land_calls": [
            {"team": t, "reason": TEL_LAND_REASONS.get(t, "")}
            for t in TEL_LAND_CALL
        ]
    })



def validate_lock_packet(payload, team):
    errors = []

    # hedef alanları
    for f in ["hedef_merkez_X", "hedef_merkez_Y",
              "hedef_genislik", "hedef_yukseklik"]:
        if payload.get(f) in (None, "", 0):
            errors.append(f"eksik:{f}")

    return errors


def _recalc_total(team: int):
    _ensure_team(team)
    s = SCORES[team]

    total = 0.0

    # ✅ POZİTİF PUANLAR
    total += s["auto_lock"]    * 500
    total += s["manual_lock"]  * 50
    total += s["kamikaze"]     * 300
    total += s["auto_land"]    * 100
    total += s["auto_takeoff"] * 50
    total += s["video_tx"]     * 50

    # ❌ SÜRELİ CEZALAR
    total -= s["tel_err_sec"] * 0.2
    total -= s["hatali_kilit_kamikaze"] * 30
    total -= s["manual_limit"] * 10
    total -= s["hss_sec"] * 5

    # ❌ TEK SEFERLİK CEZALAR
    total -= s["area_violation"]     * 150
    total -= s["boundary_violation"] * 200
    total -= s["ihrac"]              * 75

    s["total"] = round(total, 2)
    return s["total"]

def deg2rad(d): return d * math.pi / 180.0


def rad2deg(r): return r * 180.0 / math.pi


def dest_from(lat, lon, bearing_deg, dist_m):
    R = 6371000.0
    br = deg2rad(bearing_deg)
    lat1, lon1 = deg2rad(lat), deg2rad(lon)
    d_R = dist_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d_R) + math.cos(lat1) * math.sin(d_R) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(d_R) * math.cos(lat1),
                             math.cos(d_R) - math.sin(lat1) * math.sin(lat2))
    return rad2deg(lat2), rad2deg(lon2)


@app.route("/api/giris", methods=["POST"])
def giris():
    data = request.get_json(silent=True) or {}
    kadi = data.get("kadi")
    sifre = data.get("sifre")

    # kullanıcıyı bul
    user = next((u for u in VALID_USERS if u["kadi"] == kadi and u["sifre"] == sifre), None)
    if not user:
        return ("Geçersiz kullanıcı adı veya şifre", 400)

    # rastgele token üret
    token = secrets.token_hex(16)
    ISSUED_TOKENS[token] = user["takim"]

    print(f"🔐 Login OK: {kadi} → team {user['takim']} | token={token}")

    return jsonify({
        "takim_numarasi": user["takim"],
        "token": token
    }), 200


@app.route("/api/sunucusaati", methods=["GET"])
def sunucusaati():
    return jsonify(server_now_dict()), 200


# Şeman: tam olarak kullanıcının gönderdiği alanlar
REQUIRED_FIELDS = [
    "takim_numarasi", "iha_enlem", "iha_boylam", "iha_irtifa",
    "iha_dikilme", "iha_yonelme", "iha_yatis", "iha_hiz",
    "iha_batarya", "iha_otonom", "iha_kilitlenme", "gps_saati"
]
LOCK_FIELDS = ["hedef_merkez_X", "hedef_merkez_Y", "hedef_genislik", "hedef_yukseklik"]


def validate_telemetry(t):
    try:
        for f in REQUIRED_FIELDS:
            if f not in t: return False
        if int(t["iha_kilitlenme"]) == 1:
            for f in LOCK_FIELDS:
                if f not in t: return False

        # Aralık kontrolleri (makul sınırlar)
        if not (-90 <= float(t["iha_enlem"]) <= 90): return False
        if not (-180 <= float(t["iha_boylam"]) <= 180): return False
        if not (0 <= float(t["iha_irtifa"]) <= 10000): return False
        if not (-90 <= float(t["iha_dikilme"]) <= 90): return False
        if not (0 <= float(t["iha_yonelme"]) <= 360): return False
        if not (-90 <= float(t["iha_yatis"]) <= 90): return False
        if not (0 <= float(t["iha_hiz"]) <= 200): return False
        if not (0 <= int(t["iha_batarya"]) <= 100): return False
        if int(t["iha_otonom"]) not in (0, 1): return False
        if int(t["iha_kilitlenme"]) not in (0, 1): return False

        gps = t["gps_saati"]
        for k in ("saat", "dakika", "saniye", "milisaniye"):
            if k not in gps: return False
        if not (0 <= int(gps["saat"]) < 24): return False
        if not (0 <= int(gps["dakika"]) < 60): return False
        if not (0 <= int(gps["saniye"]) < 60): return False
        if not (0 <= int(gps["milisaniye"]) < 1000): return False
        return True
    except Exception:
        return False


_enemy_angle = 0.0


@app.route("/api/hss_send_flag", methods=["POST"])
def hss_send_flag():
    global HSS_SEND_ENABLED
    if not ok_auth():
        return "401", 401

    d = request.get_json(silent=True) or {}
    # enabled true/false bekliyoruz
    HSS_SEND_ENABLED = bool(d.get("enabled"))
    print("🔁 HSS_SEND_ENABLED =", HSS_SEND_ENABLED)
    return jsonify({"ok": True, "enabled": HSS_SEND_ENABLED}), 200


@app.route("/api/hss_toggle", methods=["POST"])
def hss_toggle():
    """HSS sistemini aktif/pasif yap (uçak tarafında kaçınma aktif/pasif)"""
    global HSS_SYSTEM_ACTIVE
    if not ok_auth():
        return "401", 401

    d = request.get_json(silent=True) or {}

    # "active": true/false ile kontrol
    if "active" in d:
        HSS_SYSTEM_ACTIVE = bool(d["active"])
    else:
        # Toggle
        HSS_SYSTEM_ACTIVE = not HSS_SYSTEM_ACTIVE

    status = "AKTİF ✅" if HSS_SYSTEM_ACTIVE else "PASİF ❌"
    print(f"🔄 HSS SİSTEMİ: {status}")

    # SocketIO ile tüm clientlara bildir
    socketio.emit("hss_system_status", {"hss_aktif": HSS_SYSTEM_ACTIVE})

    return jsonify({
        "ok": True,
        "hss_aktif": HSS_SYSTEM_ACTIVE,
        "message": f"HSS sistemi {status}"
    }), 200


@app.route("/api/telemetri_gonder", methods=["POST"])
def telemetri():
    # TEAM_NO artık yayın/ayrım için kullanılmıyor; projede başka yerde kullanıyorsanız kalsın.
    global _latest_telemetry, TEAM_NO, _TELEMETRY_STALE_SEC, _last_telemetry_ts, _RATE_PERIOD

    if not ok_auth():
        return "401", 401

    t = request.get_json(silent=True) or {}
    print("📡 Gelen Telemetri:", t)

    # Şema/alan kontrolü: başarısızsa 204 (gövde yok)
    if not validate_telemetry(t):
        return "", 204

    # 0) Takım id'yi GÖNDERENDEN al
    try:
        takim = int(t["takim_numarasi"])


        # ✅ takım ilk kez bağlandıysa skor tablosuna 0'la düşsün
        new_team = (takim not in SCORES)
        _ensure_team(takim)
        # ilk kez gördüysek UI'a hemen yolla (0 puanla görünsün)
        if new_team:
            _recalc_total(takim)  # total = 0.0
            emit_score_update()
    except Exception:
        now = time.time()
        TEL_LAST_SEEN[takim] = now
        return ("bad request", 400)

    # 1) Rate limit (takım bazlı) — 2 Hz (0.5 s)
    now_m = time.monotonic()
    last = _last_telemetry_ts.get(takim, 0)
    if now_m - last < _RATE_PERIOD:
        return ("3", 400)  # hızlı gönderim
    _last_telemetry_ts[takim] = now_m
    LAST_ACCEPTED_TELEMETRY_REAL_TS[takim] = time.time()

    # 2) In-memory son telemetri kaydı (takım bazlı)
    try:
        _latest_telemetry[takim] = {"telemetry": t, "ts": time.time()}
    except Exception as e:
        print("❌ _latest_telemetry güncelleme hatası:", e)

    # 3) (Opsiyonel) Geofence kontrolü — gönderene uygula
    try:
        lat = float(t.get("iha_enlem"))
        lon = float(t.get("iha_boylam"))

        GEOFENCE_LAST[takim] = time.time()  # ✅ opsiyonel: en son ne zaman kontrol edildi

        if not is_inside_fences(lat, lon):
            GEOFENCE_OUTSIDE[takim] = True  # ✅ skor loop bunu okuyacak

            socketio.emit("geofence_violation", {
                "takim": takim,
                "lat": lat,
                "lon": lon,
                "utc": now_iso()
            })
        else:
            GEOFENCE_OUTSIDE[takim] = False  # ✅ içeri döndü
            socketio.emit("geofence_ok", {"takim": takim, "utc": now_iso()})
    except Exception as e:
        print("⚠️ Geofence kontrol hatası:", e)

    # 3b) (Opsiyonel) HSS kontrolü — gönderene uygula
    try:
        lat = float(t.get("iha_enlem"))
        lon = float(t.get("iha_boylam"))

        inside = bool(HSS_SEND_ENABLED and HSS_SYSTEM_ACTIVE and is_inside_hss(lat, lon))

        prev_inside = HSS_INSIDE.get(takim, False)
        HSS_INSIDE[takim] = inside

        # ✅ HSS uyarısını anında göster (cezadan bağımsız)
        # İçerideyken 2Hz telemetriyle sürekli "hss_inside" gelir → UI watchdog düşmez.
        if inside:
            socketio.emit("hss_inside", {"takim": takim, "utc": now_iso()})
        elif prev_inside:
            # sadece çıkış anında bir kere gönder
            socketio.emit("hss_ok", {"takim": takim, "utc": now_iso()})
    except Exception as e:
        print("⚠️ HSS kontrol hatası:", e)
        HSS_INSIDE[takim] = False
        socketio.emit("hss_ok", {"takim": takim, "utc": now_iso()})

    # 4) (Opsiyonel) DB'ye yaz
    try:
        save_telemetry_row(takim, t)
    except Exception as e:
        print("save_telemetry_row hata:", e)

    # 5) Enemies: tüm güncel telemetriler (kendisi dahil)
    enemies = []
    try:
        now_ts = time.time()
        for tnum, info in list(_latest_telemetry.items()):
            ts = info.get("ts", 0)
            if _TELEMETRY_STALE_SEC is not None and (now_ts - ts) > _TELEMETRY_STALE_SEC:
                continue

            packet = info.get("telemetry", {}) or {}

            def g(k, alts=()):
                if k in packet:
                    return packet[k]
                for a in alts:
                    if a in packet:
                        return packet[a]
                return None

            enemies.append({
                "takim_numarasi": int(tnum),
                "iha_enlem": g("iha_enlem", ["enlem", "lat", "latitude"]),
                "iha_boylam": g("iha_boylam", ["boylam", "lon", "longitude"]),
                "iha_irtifa": g("iha_irtifa", ["irtifa", "alt", "altitude"]),
                "iha_dikilme": g("iha_dikilme", ["dikilme", "pitch"]),
                "iha_yonelme": g("iha_yonelme", ["yonelme", "yaw", "heading"]),
                "iha_yatis": g("iha_yatis", ["yatis", "roll"]),
                "iha_hizi": g("iha_hizi", ["iha_hiz", "hiz", "speed"]),
                "zaman_farki": int((now_ts - ts) * 1000)
            })
    except Exception as e:
        print("enemies oluştururken hata:", e)
        enemies = []

    # =========================
    # 🔒 KİLİT TAKİP GÜNCELLEME
    # =========================
    try:
        update_lock_tracking(takim, t, enemies)
    except Exception as e:
        print("⚠️ update_lock_tracking hata:", e)


    # ---- PUAN: telemetri izleme state'i (emit'ten önce) ----
    now = time.time()
    _ensure_team(takim)
    TEL_LAST_SEEN[takim] = now

    # 🔥 TELEMETRİ GERİ GELDİYSE MODE RESET
    state = TEL_STATE.get(takim)
    if state and state["mode"] == "kesik":
        TEL_STATE[takim] = {
            "mode": "ok",
            "start": now,
            "ihrac_written": False
        }

    sig = telemetry_signature(t)

    if TEL_LAST_HASH.get(takim) != sig:
        TEL_LAST_HASH[takim] = sig
        TEL_LAST_CHANGE[takim] = now
    else:
        # hash değişmediyse, change time güncellenmez (sabit demek)
        if takim not in TEL_LAST_CHANGE:
            TEL_LAST_CHANGE[takim] = now

    # =========================
    #  OTONOM KALKIŞ / İNİŞ TAKİBİ
    # =========================

    if takim not in FLIGHT_STATE:
        FLIGHT_STATE[takim] = {"airborne": False}

    state = FLIGHT_STATE[takim]

    try:
        alt = float(t.get("iha_irtifa", 0))
        otonom = int(t.get("iha_otonom", 0))
    except:
        alt = 0
        otonom = 0

    # YERDE → HAVADA
    if not state.get("airborne", False) and alt > 20:
        state["airborne"] = True
        if SCORING_ACTIVE and otonom == 1:
            print("✅ OTONOM KALKIŞ")
            SCORES[takim]["auto_takeoff"] += 1
            _recalc_total(takim)

    elif state.get("airborne", False) and alt < 5:
        state["airborne"] = False
        if SCORING_ACTIVE and otonom == 1:
            print("✅ OTONOM İNİŞ")
            SCORES[takim]["auto_land"] += 1
            _recalc_total(takim)

    FLIGHT_STATE[takim] = state


    # 6) skor güncelle
    try:
        _recalc_total(takim)
        emit_score_update()
    except Exception as _e:
        print("score update hata:", _e)

    # 7) HTTP cevabı aynı formatta (UI geriye uyumlu)
    return jsonify({
        "sunucusaati": server_now_dict(),
        "konumBilgileri": enemies,
        "hss_koordinat_bilgileri": [
            {"id": it["id"], "hssEnlem": it["lat"], "hssBoylam": it["lon"], "hssYaricap": it["radius"]}
            for it in list_hss()
        ] if HSS_SYSTEM_ACTIVE else []
    }), 200


def _invalid_lock(team):
    _ensure_team(team)

    if SCORING_ACTIVE:
        SCORES[team]["hatali_kilit_kamikaze"] += 1  # -30
        _recalc_total(team)
        emit_score_update()

    socketio.emit("lock_event", {
        "takim": team,
        "kilit_bitis_gps": None,
        "otonom_kilitlenme": None
    })


import time
import math

ACTIVE_LOCK_START = {}        # team -> lock_start_time
CURRENT_TARGET = {}           # team -> target_id
LAST_LOCKED_TARGET = {}       # team -> last_success_target
LAST_LOCK_SIG = {}            # duplicate lock kontrol


def update_lock_tracking(team, t, enemies):

    kilit = t.get("iha_kilitlenme") == 1

    last_video = LAST_VIDEO_TS.get(team)
    video_ok = last_video and (time.time() - last_video) <= 1.0

    if not video_ok:
        ACTIVE_LOCK_START.pop(team, None)
        CURRENT_TARGET.pop(team, None)
        return

    hedef_var = all([
        t.get("hedef_merkez_X") is not None,
        t.get("hedef_merkez_Y") is not None,
        t.get("hedef_genislik") is not None,
        t.get("hedef_yukseklik") is not None
    ])

    # hedef kimliği (basit bounding box imzası)
    target_sig = (
        t.get("hedef_merkez_X"),
        t.get("hedef_merkez_Y"),
        t.get("hedef_genislik"),
        t.get("hedef_yukseklik")
    )

    # -------------------------
    # 1️⃣ Kilit kapalıysa reset
    # -------------------------
    if not kilit:
        ACTIVE_LOCK_START.pop(team, None)
        CURRENT_TARGET.pop(team, None)
        return

    # -------------------------
    # 2️⃣ Hedef kaybolduysa reset
    # -------------------------
    if not hedef_var:
        ACTIVE_LOCK_START.pop(team, None)
        CURRENT_TARGET.pop(team, None)
        return

    # -------------------------
    # 3️⃣ Yeni hedef başladıysa reset + başlat
    # -------------------------
    if CURRENT_TARGET.get(team) != target_sig:
        CURRENT_TARGET[team] = target_sig
        ACTIVE_LOCK_START[team] = time.time()
        return

    # -------------------------
    # 4️⃣ Aynı hedef devam ediyorsa süre akmaya devam eder
    # -------------------------
    if team not in ACTIVE_LOCK_START:
        ACTIVE_LOCK_START[team] = time.time()




@app.route("/api/kilitlenme_bilgisi", methods=["POST"])
def kilitlenme():

    if not ok_auth():
        return "401", 401

    data = request.get_json(silent=True) or {}

    # 🔐 Takım sadece token'dan
    team = auth_team()
    if team is None:
        return "401", 401

    team = int(team)
    _ensure_team(team)

    # =========================
    # 1️⃣ ZORUNLU ALAN
    # =========================
    kb = data.get("kilitlenmeBitisZamani")
    if not isinstance(kb, dict):
        _invalid_lock(team)
        return "", 204

    for f in ("saat", "dakika", "saniye", "milisaniye"):
        if f not in kb:
            _invalid_lock(team)
            return "", 204

    try:
        ok_flag = int(data.get("otonom_kilitlenme"))
    except Exception:
        _invalid_lock(team)
        return "", 204

    if ok_flag not in (0, 1):
        _invalid_lock(team)
        return "", 204

    # =========================
    # 2️⃣ TELEMETRİ GERÇEK KİLİT VAR MI?
    # =========================
    latest = _latest_telemetry.get(team, {}).get("telemetry", {})

    if latest.get("iha_kilitlenme") != 1:
        print("❌ Telemetride kilit aktif değil")
        _invalid_lock(team)
        return "", 204

    # hedef bilgileri zorunlu
    if not all([
        latest.get("hedef_merkez_X"),
        latest.get("hedef_merkez_Y"),
        latest.get("hedef_genislik"),
        latest.get("hedef_yukseklik")
    ]):
        print("❌ Hedef X/Y/W/H eksik")
        _invalid_lock(team)
        return "", 204

    # video son 1 saniyede gelmiş mi
    #last_video = LAST_VIDEO_TS.get(team)
    #if not last_video or (time.time() - last_video) > 1.0:
    #    print(f"❌ Takım {team}: son 1 saniyede görüntü yok, kilit geçersiz")
    #    _invalid_lock(team)
    #    return "", 204
#
    # =========================
    # 3️⃣ 4 SANİYE TAKİP KONTROLÜ
    # =========================
    start_time = ACTIVE_LOCK_START.get(team)
    #if not start_time:
    #    print("❌ Lock start yok")
    #    _invalid_lock(team)
    #    return "", 204

    #duration = time.time() - start_time
    #if duration < 4:
    #    print("❌ 4 saniye dolmadı:", duration)
    #    _invalid_lock(team)
    #    return "", 204

    # =========================
    # 4️⃣ HEDEF TEKRAR KONTROLÜ
    # =========================
    target = CURRENT_TARGET.get(team)
    #if target is None:
    #    print("❌ Target yok")
    #    _invalid_lock(team)
    #    return "", 204

    #if LAST_LOCKED_TARGET.get(team) == target:
    #    print("❌ Aynı hedefe tekrar kilit")
    #    _invalid_lock(team)
    #    return "", 204

    # =========================
    # 5️⃣ DUPLICATE ZAMAN
    # =========================
    lock_sig = (
        int(kb["saat"]),
        int(kb["dakika"]),
        int(kb["saniye"]),
        int(kb["milisaniye"])
    )

    if LAST_LOCK_SIG.get(team) == lock_sig:
        print("❌ Duplicate paket")
        _invalid_lock(team)
        return "", 204

    LAST_LOCK_SIG[team] = lock_sig

    # =========================
    # 6️⃣ PUANLAMA
    # =========================
    if SCORING_ACTIVE:
        if ok_flag == 1:
            SCORES[team]["auto_lock"] += 1
        else:
            SCORES[team]["manual_lock"] += 1

        _recalc_total(team)
        emit_score_update()

    LAST_LOCKED_TARGET[team] = target

    # =========================
    # 7️⃣ DB
    # =========================
    try:
        save_lock_row(team, data)
    except Exception as e:
        print("❌ save_lock_row:", e)

    # =========================
    # 8️⃣ UI
    # =========================
    socketio.emit("lock_event", {
        "takim": team,
        "kilit_bitis_gps": kb,
        "otonom_kilitlenme": ok_flag
    })

    # reset
    ACTIVE_LOCK_START.pop(team, None)
    CURRENT_TARGET.pop(team, None)

    return "OK", 200


@app.route("/api/kamikaze_bilgisi", methods=["POST"])
def kamikaze():
    if not ok_auth():
        return "401", 401

    d = request.get_json(silent=True) or {}

    # 🔐 TEAM MUTLAKA TOKEN'DAN
    team = auth_team()
    if team is None:
        team = d.get("kaynak_takim")

    if team is None:
        print("❌ TEAM BULUNAMADI")
        return "", 204

    team = int(team)
    _ensure_team(team)
    d["kaynak_takim"] = team

    kb = d.get("kamikazeBaslangicZamani", {})
    ke = d.get("kamikazeBitisZamani", {})

    # ❌ HATALI PAKET
    if not (
        all(k in kb for k in ("saat", "dakika", "saniye", "milisaniye")) and
        all(k in ke for k in ("saat", "dakika", "saniye", "milisaniye")) and
        d.get("qrMetni")
    ):
        if SCORING_ACTIVE:
            SCORES[team]["hatali_kilit_kamikaze"] += 1
            _recalc_total(team)
            emit_score_update()

        print("❌ HATALI KAMİKAZE -30 yazıldı")
        return "", 204

    # ✅ DOĞRU PAKET
    try:
        save_kamikaze_row(d)
        print("📝 kamikaze→DB OK")
    except Exception as e:
        print("❌ save_kamikaze_row HATA:", e)

    if SCORING_ACTIVE:
        SCORES[team]["kamikaze"] += 1
        _recalc_total(team)
        emit_score_update()

    socketio.emit('kamikaze_event', {
        "kaynak_takim": team,
        "qrMetni": d.get("qrMetni"),
        "kamikazeBaslangicZamani": kb,
        "kamikazeBitisZamani": ke,
        "sunucusaati": server_now_dict(),
    })

    print("💥 Kamikaze:", d)
    return "OK", 200


@app.route("/api/qr_koordinati", methods=["GET"])
def qr():
    if not ok_auth():
        return "401", 401
    return jsonify({"qrEnlem": 38.701690, "qrBoylam": 27.457857}), 200


# helper:
def ok_auth_or_dev():
    # URL parametresi ile dev mod açılırsa yetkisiz erişime izin ver
    if request.args.get("dev") == "1":
        return True
    return ok_auth()


def ok_auth_or_public_hss():
    """
    HSS endpointi, yki_gorev_yazilimi'ndaki fetch_hss() ile
    (çoğu zaman Authorization/cookie olmadan) çağrılabildiği için
    auth YOKSA bile geçelim. Auth varsa yine kabul.
    """
    return ok_auth() or True  # HSS'i public yap


@app.route("/api/hss", methods=["GET"])
def api_hss_list():
    return jsonify({"ok": True, "items": list_hss()}), 200


@app.route("/api/hss", methods=["POST"])
def api_hss_create():
    if not ok_auth(): return "401", 401
    d = request.get_json(silent=True) or {}
    name = (d.get("name") or "").strip() or "HSS"
    lat = d.get("lat")
    lon = d.get("lon")
    radius = d.get("radius")
    try:
        _id = insert_hss(name, float(lat), float(lon), float(radius))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    try:
        socketio.emit("hss_update", {"items": list_hss()})
    except:
        pass
    return jsonify({"ok": True, "id": _id}), 200


@app.route("/api/hss/<int:hid>", methods=["PUT"])
def api_hss_update(hid):
    if not ok_auth(): return "401", 401
    d = request.get_json(silent=True) or {}
    try:
        update_hss(hid, **d)
        socketio.emit("hss_update", {"items": list_hss()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True}), 200


@app.route("/api/hss/<int:hid>", methods=["DELETE"])
def api_hss_delete(hid):
    if not ok_auth(): return "401", 401
    delete_hss(hid)
    try:
        socketio.emit("hss_update", {"items": list_hss()})
    except:
        pass
    return jsonify({"ok": True}), 200


@app.route("/api/hss_koordinatlari", methods=["GET"])
def hss_public():
    if not HSS_SEND_ENABLED:
        hss_list = []
    else:
        items = list_hss()
        hss_list = [
            {"id": it["id"], "hssEnlem": it["lat"], "hssBoylam": it["lon"], "hssYaricap": it["radius"]}
            for it in items
        ]

    return jsonify({
        "sunucusaati": server_now_dict(),
        "hss_koordinat_bilgileri": hss_list,
        "durum": "aktif" if hss_list else "bos",
        "hss_aktif": HSS_SYSTEM_ACTIVE
    }), 200



@app.route("/api/fences", methods=["GET"])
def get_fences():
    return jsonify({"ok": True, "items": list_fences()}), 200


@app.route("/api/fences", methods=["POST"])
def create_fence():
    if not ok_auth(): return "401", 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "Geofence"
    kind = data.get("kind")
    gj = data.get("geojson")
    color = data.get("color") or "#ef4444"
    if kind not in ("polygon",) or not gj:
        return jsonify({"ok": False, "error": "invalid payload"}), 400
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("INSERT INTO fences(name,kind,geojson,color,updated_at) VALUES(?,?,?,?,?)",
                (name, kind, json.dumps(gj), color, now_iso()))
    con.commit();
    con.close()
    items = list_fences()
    socketio.emit("fences_update", {"items": items})
    return jsonify({"ok": True, "items": items}), 200


@app.route("/api/fences/<int:fid>", methods=["PUT"])
def update_fence(fid):
    if not ok_auth(): return "401", 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "Geofence"
    kind = data.get("kind")
    gj = data.get("geojson")
    color = data.get("color") or "#ef4444"
    if kind not in ("polygon",) or not gj:
        return jsonify({"ok": False, "error": "invalid payload"}), 400
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("UPDATE fences SET name=?,kind=?,geojson=?,color=?,updated_at=? WHERE id=?",
                (name, kind, json.dumps(gj), color, now_iso(), fid))
    con.commit();
    con.close()
    items = list_fences()
    socketio.emit("fences_update", {"items": items})
    return jsonify({"ok": True, "items": items}), 200


@app.route("/api/fences/<int:fid>", methods=["DELETE"])
def delete_fence(fid):
    if not ok_auth(): return "401", 401
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("DELETE FROM fences WHERE id=?", (fid,))
    con.commit();
    con.close()
    items = list_fences()
    socketio.emit("fences_update", {"items": items})
    return jsonify({"ok": True, "items": items}), 200

from flask import send_from_directory
import os

@app.route("/video_wall_static")
def video_wall_static():
    return send_from_directory(
        os.path.dirname(__file__),
        "video_wall_static.html"
    )

@app.route("/history")
def history():
    return send_file("history.html")

@app.route("/api/scoring_toggle", methods=["POST"])
def api_scoring_toggle():
    global SCORING_ACTIVE

    data = request.get_json(silent=True) or {}
    SCORING_ACTIVE = bool(data.get("active", False))

    socketio.emit("scoring_status", {
        "scoring_active": SCORING_ACTIVE
    })

    return jsonify({
        "ok": True,
        "scoring_active": SCORING_ACTIVE
    }), 200

@app.route("/api/scoring_reset", methods=["POST"])
def api_scoring_reset():
    global TEL_LAND_CALL, TEL_LAND_REASONS, SCORING_ACTIVE
    global TEL_STATE, TEL_LAST_SEEN, TEL_LAST_CHANGE, TEL_LAST_HASH
    global TEAM_1HZ_VIOLATION, LAST_ACCEPTED_TELEMETRY_REAL_TS
    global HSS_INSIDE, HSS_WARNED, GEOFENCE_OUTSIDE, GEOFENCE_WARNED_AT
    global GEOFENCE_IHRAC_DONE, GEOFENCE_LAST, IHRAC_DONE, FLIGHT_STATE
    global VIDEO_OK, LAST_VIDEO_TS

    for team in list(SCORES.keys()):
        SCORES[team]["auto_lock"] = 0
        SCORES[team]["manual_lock"] = 0
        SCORES[team]["kamikaze"] = 0
        SCORES[team]["auto_land"] = 0
        SCORES[team]["auto_takeoff"] = 0
        SCORES[team]["video_tx"] = 0

        SCORES[team]["tel_err_sec"] = 0
        SCORES[team]["hatali_kilit_kamikaze"] = 0
        SCORES[team]["manual_limit"] = 0
        SCORES[team]["hss_sec"] = 0

        SCORES[team]["area_violation"] = 0
        SCORES[team]["boundary_violation"] = 0
        SCORES[team]["ihrac"] = 0
        SCORES[team]["total"] = 0.0

    TEL_LAND_CALL = set()
    TEL_LAND_REASONS = {}

    TEL_STATE = {}
    TEL_LAST_SEEN = {}
    TEL_LAST_CHANGE = {}
    TEL_LAST_HASH = {}

    TEAM_1HZ_VIOLATION = {}
    LAST_ACCEPTED_TELEMETRY_REAL_TS = {}

    HSS_INSIDE = {}
    HSS_WARNED = set()

    GEOFENCE_OUTSIDE = {}
    GEOFENCE_WARNED_AT = {}
    GEOFENCE_IHRAC_DONE = set()
    GEOFENCE_LAST = {}

    IHRAC_DONE = set()
    FLIGHT_STATE = {}

    VIDEO_OK = {}
    LAST_VIDEO_TS = {}

    SCORING_ACTIVE = False

    socketio.emit("scoring_status", {
        "scoring_active": SCORING_ACTIVE
    })

    emit_score_update()

    return jsonify({
        "ok": True,
        "message": "Skorlar ve sayaçlar sıfırlandı"
    }), 200

@app.route("/api/scoring_status", methods=["GET"])
def api_scoring_status():
    return jsonify({
        "ok": True,
        "scoring_active": SCORING_ACTIVE
    }), 200



def point_in_polygon(lat, lon, polygon_latlon):
    # polygon_latlon: [[lat,lon], ...] (ilk/son kapanmasa da işler)
    x, y = lon, lat
    inside = False
    pts = [(p[1], p[0]) for p in polygon_latlon]  # (x,y) = (lon,lat)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def is_inside_fences(lat, lon):
    for f in list_fences():
        gj = f["geojson"]
        if f["kind"] == "polygon":
            try:
                outer = gj["geometry"]["coordinates"][0]  # [[lon,lat],...]
                poly = [[pt[1], pt[0]] for pt in outer]
                if point_in_polygon(lat, lon, poly): return True
            except:
                pass
    return False


def score_tick_loop():
    while True:
        socketio.sleep(1)
        now = time.time()

        for team in list(TEL_LAST_SEEN.keys()):
            _ensure_team(team)

            last_seen = TEL_LAST_SEEN.get(team, 0)
            # ✅ OFFLINE ise: hiçbir ceza ilerlemesin HSS
            offline = (now - last_seen) > OFFLINE_AFTER_SEC
            if offline:
                # HSS / geofence durur
                GEOFENCE_OUTSIDE[team] = False
                GEOFENCE_WARNED_AT.pop(team, None)
                HSS_INSIDE[team] = False
                _recalc_total(team)
                continue

            if SCORING_ACTIVE:
                # HSS cezası
                if HSS_SYSTEM_ACTIVE and HSS_INSIDE.get(team, False):
                    SCORES[team]["hss_sec"] += 1
                    HSS_WARNED.add(team)

                    if SCORES[team]["hss_sec"] >= 30:
                        land_call(team, "HSS ihlali (30 sn)")
                else:
                    HSS_WARNED.discard(team)

                # Geofence cezası
                if GEOFENCE_OUTSIDE.get(team, False):
                    if team not in GEOFENCE_WARNED_AT:
                        GEOFENCE_WARNED_AT[team] = now

                    if (now - GEOFENCE_WARNED_AT[team]) >= 10.0:
                        if team not in GEOFENCE_IHRAC_DONE:
                            SCORES[team]["boundary_violation"] += 1
                            GEOFENCE_IHRAC_DONE.add(team)
                            land_call(team, "Sınır dışı ihlali (10 sn)")
                else:
                    GEOFENCE_WARNED_AT.pop(team, None)
                    GEOFENCE_IHRAC_DONE.discard(team)

                _recalc_total(team)
            else:
                # Puanlama kapalıysa sadece uyarı state'leri temiz kalsın
                HSS_WARNED.discard(team)

        emit_score_update()

def telemetry_watchdog():
    while True:
        socketio.sleep(1)
        now = time.time()

        for team in list(TEL_LAST_SEEN.keys()):
            _ensure_team(team)

            last_seen = TEL_LAST_SEEN.get(team, 0)
            offline = (now - last_seen) > OFFLINE_AFTER_SEC

            last_change = TEL_LAST_CHANGE.get(team, now)
            sabit = (now - last_change) >= 1

            state = TEL_STATE.get(team, {
                "mode": "ok",
                "start": now,
                "ihrac_written": False
            })

            # 1) TELEMETRİ KESİK
            if offline:
                if state["mode"] != "kesik":
                    state = {
                        "mode": "kesik",
                        "start": now,
                        "ihrac_written": False
                    }

                elapsed = now - state["start"]

                if SCORING_ACTIVE:
                    SCORES[team]["tel_err_sec"] = int(elapsed)

                    if elapsed >= 10 and not state["ihrac_written"]:
                        print(f"İHRAÇ (KESİK): Takım {team}")
                        SCORES[team]["ihrac"] += 1
                        land_call(team, "Telemetri kesildi (10 sn)")
                        state["ihrac_written"] = True
                        _recalc_total(team)

            # 2) TELEMETRİ SABİT
            elif sabit:
                if state["mode"] != "sabit":
                    state = {
                        "mode": "sabit",
                        "start": now,
                        "ihrac_written": False
                    }

                elapsed = now - state["start"]

                if SCORING_ACTIVE:
                    SCORES[team]["tel_err_sec"] = int(elapsed)

                    if elapsed >= 10 and not state["ihrac_written"]:
                        print(f"İHRAÇ (SABİT): Takım {team}")
                        SCORES[team]["ihrac"] += 1
                        land_call(team, "Telemetri sabit (10 sn)")
                        state["ihrac_written"] = True
                        _recalc_total(team)

            # 3) HER ŞEY NORMAL
            else:
                state = {
                    "mode": "ok",
                    "start": now,
                    "ihrac_written": False
                }

                if SCORING_ACTIVE:
                    SCORES[team]["tel_err_sec"] = 0

            TEL_STATE[team] = state

        emit_score_update()

def telemetry_broadcast_loop():
    while True:
        try:
            now = time.time()

            # 1 Hz altı kontrolü
            for team, last_ts in list(LAST_ACCEPTED_TELEMETRY_REAL_TS.items()):
                late = (now - last_ts) > MIN_TELEMETRY_PERIOD
                old = TEAM_1HZ_VIOLATION.get(team, False)

                if late != old:
                    TEAM_1HZ_VIOLATION[team] = late
                    socketio.emit("telemetry_rate_status", {
                        "takim": team,
                        "min_1hz_ok": (not late),
                        "last_age_sec": round(now - last_ts, 3)
                    })

            # Tüm takımların son telemetrisini 1 Hz yayınla
            for team, item in list(_latest_telemetry.items()):
                telem = item.get("telemetry", {})

                enemies = []
                for other_team, other_item in list(_latest_telemetry.items()):
                    if other_team == team:
                        continue

                    if now - other_item.get("ts", 0) <= _TELEMETRY_STALE_SEC:
                        packet = other_item.get("telemetry", {}) or {}

                        def g(k, alts=()):
                            if k in packet:
                                return packet[k]
                            for a in alts:
                                if a in packet:
                                    return packet[a]
                            return None

                        enemies.append({
                            "takim_numarasi": int(other_team),
                            "iha_enlem": g("iha_enlem", ["enlem", "lat", "latitude"]),
                            "iha_boylam": g("iha_boylam", ["boylam", "lon", "longitude"]),
                            "iha_irtifa": g("iha_irtifa", ["irtifa", "alt", "altitude"]),
                            "iha_dikilme": g("iha_dikilme", ["dikilme", "pitch"]),
                            "iha_yonelme": g("iha_yonelme", ["yonelme", "yaw", "heading"]),
                            "iha_yatis": g("iha_yatis", ["yatis", "roll"]),
                            "iha_hiz": g("iha_hiz", ["hiz", "speed"]),
                            "zaman_farki": int((now - other_item.get("ts", 0)) * 1000)
                        })

                socketio.emit("telemetry_update", {
                    "takim": team,
                    "telemetry": telem,
                    "sunucusaati": server_now_dict(),
                    "enemies": enemies
                })

        except Exception as e:
            print("❌ telemetry_broadcast_loop hata:", e)

        socketio.sleep(1.0)


@socketio.on('connect')
def on_connect():
    print("🔌 socket connected:", request.sid)


@socketio.on('disconnect')
def on_disconnect():
    print("🔌 socket disconnected:", request.sid)


from flask_socketio import emit
from flask import request


@socketio.on('fetch_history')
def on_fetch_history(payload):
    """
    payload: { takim?: int|string, start?: 'YYYY-MM-DDTHH:MM:SSZ', end?: 'YYYY-MM-DDTHH:MM:SSZ', limit?: int }
    """
    try:
        takim = payload.get("takim") if isinstance(payload, dict) else None
        start = payload.get("start") if isinstance(payload, dict) else None
        end = payload.get("end") if isinstance(payload, dict) else None
        limit = payload.get("limit") if isinstance(payload, dict) else 1000

        rows = query_telemetry_history(takim, start, end, limit)
        emit('history_result', {"ok": True, "rows": rows, "count": len(rows)}, room=request.sid)
    except Exception as e:
        print("❌ fetch_history hata:", e)
        emit('history_result', {"ok": False, "error": str(e)}, room=request.sid)


@socketio.on('fetch_locks')
def on_fetch_locks(payload):
    try:
        takim = payload.get("kaynak") if isinstance(payload, dict) else None
        start = payload.get("start") if isinstance(payload, dict) else None
        end = payload.get("end") if isinstance(payload, dict) else None
        limit = payload.get("limit") if isinstance(payload, dict) else 1000

        rows = query_locks_history(takim, start, end, limit)

        emit('locks_result',
             {"ok": True, "rows": rows, "count": len(rows)},
             room=request.sid)

    except Exception as e:
        print("❌ fetch_locks hata:", e)
        emit('locks_result',
             {"ok": False, "error": str(e)},
             room=request.sid)


@socketio.on('fetch_kamikaze')
def on_fetch_kamikaze(payload):
    """
    payload: { kaynak?: int|string, start?: 'YYYY-MM-DDTHH:MM:SSZ', end?: 'YYYY-MM-DDTHH:MM:SSZ', limit?: int }
    """
    try:
        kaynak = payload.get("kaynak") if isinstance(payload, dict) else None
        start = payload.get("start") if isinstance(payload, dict) else None
        end = payload.get("end") if isinstance(payload, dict) else None
        limit = payload.get("limit") if isinstance(payload, dict) else 1000

        rows = query_kamikaze_history(kaynak, start, end, limit)
        emit('kamikaze_result', {"ok": True, "rows": rows, "count": len(rows)}, room=request.sid)
    except Exception as e:
        print("❌ fetch_kamikaze hata:", e)
        emit('kamikaze_result', {"ok": False, "error": str(e)}, room=request.sid)

if __name__ == "__main__":

    print("DB PATH =", os.path.abspath("iha_logs.db"))
    init_db()
    socketio.start_background_task(score_tick_loop)
    socketio.start_background_task(telemetry_watchdog)
    socketio.start_background_task(telemetry_broadcast_loop)
    socketio.start_background_task(video_probe_loop)
    socketio.run(app, host="0.0.0.0", port=10001, debug=False, use_reloader=False)

