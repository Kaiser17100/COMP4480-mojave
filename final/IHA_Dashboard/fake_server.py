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
#  STREAM MANAGER  ?  Native ffplay pencereleri
# =============================================
# Her tak?m i�in ayr? bir ffplay penceresi a�?l?r.
# Do?rudan UDP ? decode ? ekran. S?f?r ara katman, s?f?r kasma.

STREAM_PROCS = {}  # takim_id -> { "proc": Popen, "ip": str, "port": int }
STREAM_LOCK = threading.Lock()

LAST_VIDEO_TS = {}  # team -> son g�r�nt� zaman?
VIDEO_OK = {}  # team -> son probe'da g�r�nt� var m?

SCORING_ACTIVE = False  # ba?lang?�ta kapal?

# Telemetri / kilitlenme / kamikaze verilerini DB'ye kaydetme.
# False iken dashboard anl?k �al???r; sadece ge�mi? kay?t olu?turulmaz.
SAVE_MATCH_DATA_TO_DB = False

# Performans modu: telemetri geçmişi/logları kapalı.
# Anlık dashboard ve skor çalışır; telemetri/kilit/kamikaze geçmişe yazılmaz.
LOG_TELEMETRY = False
DISABLE_HISTORY_QUERIES = True
SCORE_EMIT_MIN_PERIOD = 1.0
_LAST_SCORE_EMIT_TS = 0.0

# Varsay?lan tak?m port e?lemeleri (UI'dan ge�ersiz k?l?nabilir)
DEFAULT_STREAM_PORTS = {
    20: 5420,
    1: 5425,
    2: 5426,
    3: 5427,
    4: 5428,
    5: 5429,
}
DEFAULT_STREAM_IP = "0.0.0.0"


def _start_ffplay(takim_id, ip, port):
    """ffplay penceresini ba?lat?r. Do?rudan UDP ? ekran."""
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

    print(f"? ffplay komutu: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    import time as _time
    _time.sleep(0.3)
    if proc.poll() is not None:
        print(f"? ffplay an?nda kapand?! Exit code: {proc.returncode}")
    else:
        print(f"? ffplay �al???yor PID={proc.pid}  pencere: {window_title}")

    return proc


def _stop_ffplay(takim_id):
    """Belirtilen tak?m?n ffplay s�recini durdurur."""
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
    """Bir tak?m?n ffplay penceresini a�ar."""
    data = request.get_json(silent=True) or {}
    ip = data.get("ip", DEFAULT_STREAM_IP)
    port = data.get("port", DEFAULT_STREAM_PORTS.get(takim_id, 5400 + takim_id))

    # Zaten �al???yorsa �nce durdur
    _stop_ffplay(takim_id)

    proc = _start_ffplay(takim_id, ip, port)
    with STREAM_LOCK:
        STREAM_PROCS[takim_id] = {"proc": proc, "ip": ip, "port": port}

    print(f"??  Stream BA?LATILDI: Tak?m {takim_id}  udp://{ip}:{port}")
    return jsonify({"ok": True, "takim": takim_id, "ip": ip, "port": port}), 200


@app.route("/api/stream/stop/<int:takim_id>", methods=["POST"])
def api_stream_stop(takim_id):
    """Bir tak?m?n ffplay penceresini kapat?r."""
    _stop_ffplay(takim_id)
    print(f"??  Stream DURDURULDU: Tak?m {takim_id}")
    return jsonify({"ok": True, "takim": takim_id}), 200


@app.route("/api/stream/status", methods=["GET"])
def api_stream_status():
    """Aktif stream'lerin durumunu d�nd�r�r."""
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


def _udp_probe(ip, port, timeout=0.2):
    """
    Belirtilen UDP portunda k?sa s�re dinler, veri gelip gelmedi?ine bakar.
    Native thread'de �al??t?r?lmal? (eventlet uyumlulu?u i�in).
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
    Belirtilen UDP portunda veri olup olmad???n? kontrol eder.
    Body: { "ip": "0.0.0.0", "port": 5425 }
    """
    data = request.get_json(silent=True) or {}
    ip = data.get("ip", "0.0.0.0")
    port = data.get("port")
    if not port:
        return jsonify({"ok": False, "error": "port gerekli"}), 400

    try:
        result = _udp_probe(ip, int(port))
    except Exception as e:
        result = {"has_data": False, "bytes": 0, "error": str(e)}

    # Tak?m? bul
    team = None
    try:
        # �nce body'den geldiyse onu kullan
        if data.get("takim_id") is not None:
            team = int(data.get("takim_id"))
        else:
            # Porttan tak?m e?lemesi bul
            for tid, p in DEFAULT_STREAM_PORTS.items():
                if int(p) == int(port):
                    team = int(tid)
                    break
    except Exception as e:
        print("? team bulma hatas?:", e)

    # G�r�nt� durumu g�ncelle
    if team is not None:
        if result.get("has_data"):
            LAST_VIDEO_TS[team] = time.time()
            VIDEO_OK[team] = True

            try:
                _ensure_team(team)

                # sadece ilk kez 1 yap, s�rekli artmas?n
                if SCORING_ACTIVE and is_team_online_for_scoring(team) and SCORES[team]["video_tx"] == 0:
                    SCORES[team]["video_tx"] = 1
                    _recalc_total(team)
                    emit_score_update()
                    print(f"? G�r�nt� alg?land? ? Tak?m {team} video_tx=1, +50 puan")
            except Exception as e:
                print("? video_tx puanlama hatas?:", e)
        else:
            VIDEO_OK[team] = False

    return jsonify({"ok": True, "port": port, **result}), 200


CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")  # dev i�in *; prod?da domain k?s?tla

# Basit kimlik & oturum
VALID_USERS = [
    {"kadi": "anafarta", "sifre": "123", "takim": 25},
    {"kadi": "yem", "sifre": "123456", "takim": 20},
    {"kadi": "deneme", "sifre": "deneme", "takim": 26},
    {"kadi": "1", "sifre": "1", "takim": 1},
    {"kadi": "2", "sifre": "2", "takim": 2},
    {"kadi": "3", "sifre": "3", "takim": 3},
    {"kadi": "4", "sifre": "4", "takim": 4},
    {"kadi": "5", "sifre": "5", "takim": 5}
]

ISSUED_TOKENS = {}  # token ? team

# In-memory en son telemetri kay?tlar?: {takim_numarasi: {"telemetry": t, "ts": time.time()}}
_latest_telemetry = {}
OFFLINE_EMITTED = set()  # offline olay? ayn? tak?m i�in tekrar tekrar yay?nlanmas?n

# (Opsiyonel) ka� saniyeden eski telemetry'i d�?man listesinden �?karmak istersin
_TELEMETRY_STALE_SEC = 5.0  # �rn. 5 saniye; ger�ek testte network ko?ullar?na g�re artt?rabilirsin
TEAM_NO = None

# HSS'ler u�a?a g�nderilsin mi?
HSS_SEND_ENABLED = True

# HSS sisteminin aktif/pasif durumu (UI'den kontrol edilebilir)
HSS_SYSTEM_ACTIVE = False  # Ba?lang?�ta KAPALI
HSS_ACTIVE_SINCE = None  # HSS aktif edildi?i an; ceza 60 sn sonra ba?lar
HSS_PENALTY_GRACE_SEC = 60.0

TOKEN = "fake_token_123"
SESSION_COOKIE = "sessionid"

# 2 Hz limiti (tak?m bazl?)
_last_telemetry_ts = {}
_RATE_PERIOD = 0.35  # saniye -> 2 Hz toleranslı; jitter yüzünden paket düşmesin

MIN_TELEMETRY_PERIOD = 1.0  # tak?m en ge� 1 sn i�inde veri g�ndermeli
TEAM_1HZ_VIOLATION = {}  # team -> bool

LAST_ACCEPTED_TELEMETRY_REAL_TS = {}

# --- HSS pencere kontrol� ---
_HSS_EMPTY1_SEC = 10  # ilk 10 saniye bo?
_HSS_ACTIVE_SEC = 10  # sonraki 10 saniye dolu
_SERVER_START_MONO = time.monotonic()

# --- SQLite setup ---
import os, sqlite3, json

DB_PATH = os.path.join(os.path.dirname(__file__), "iha_logs.db")


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS telemetry
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    ts_utc
                    TEXT
                    NOT
                    NULL,
                    takim
                    INTEGER
                    NOT
                    NULL,
                    enlem
                    REAL,
                    boylam
                    REAL,
                    irtifa
                    REAL,
                    hiz
                    REAL,
                    batarya
                    INTEGER,
                    raw_json
                    TEXT
                );
                """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_takim_ts ON telemetry(takim, ts_utc);")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS locks
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    takim
                    INTEGER
                    NOT
                    NULL,
                    saat
                    INTEGER
                    NOT
                    NULL,
                    dakika
                    INTEGER
                    NOT
                    NULL,
                    saniye
                    INTEGER
                    NOT
                    NULL,
                    milisaniye
                    INTEGER
                    NOT
                    NULL,
                    otonom_kilitlenme
                    INTEGER
                    NOT
                    NULL,
                    ts_utc
                    TEXT
                    NOT
                    NULL
                );
                """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_locks_ts ON locks(ts_utc);")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS kamikaze
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    ts_utc
                    TEXT
                    NOT
                    NULL,    -- ISO8601 UTC (server zaman?)
                    kaynak_takim
                    INTEGER, -- g�nderende varsa
                    qr_metni
                    TEXT
                    NOT
                    NULL,
                    baslangic_gps
                    TEXT
                    NOT
                    NULL,    -- JSON string
                    bitis_gps
                    TEXT
                    NOT
                    NULL,    -- JSON string
                    extra_json
                    TEXT     -- ham paket / ek alanlar
                )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kamikaze_ts ON kamikaze(ts_utc)")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS fences
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    name
                    TEXT,
                    kind
                    TEXT
                    NOT
                    NULL,      -- 'polygon' (dikd�rtgeni GeoJSON Polygon tutaca??z)
                    geojson
                    TEXT
                    NOT
                    NULL,      -- GeoJSON Feature
                    color
                    TEXT
                    DEFAULT
                    '#ef4444', -- k?rm?z?
                    updated_at
                    TEXT
                    NOT
                    NULL
                );
                """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fences_updated ON fences(updated_at);")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS hss
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    name
                    TEXT,
                    lat
                    REAL
                    NOT
                    NULL,
                    lon
                    REAL
                    NOT
                    NULL,
                    radius
                    REAL
                    NOT
                    NULL, -- metre
                    active
                    INTEGER
                    NOT
                    NULL
                    DEFAULT
                    1,
                    updated_at
                    TEXT
                    NOT
                    NULL
                );
                """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hss_active ON hss(active);")

    con.commit()
    con.close()
    print("? DB haz?r:", DB_PATH)


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# HSS ve geofence verileri telemetri geldikçe değişmediği için DB'den her pakette okunmaz.
# Bu cache, 5+ uçak aynı anda bağlıyken SQLite bağlantı yükünü ciddi azaltır.
_SPATIAL_CACHE_LOCK = threading.RLock()
_FENCES_CACHE = None
_HSS_CACHE = None


def list_fences(force_refresh=False):
    global _FENCES_CACHE
    with _SPATIAL_CACHE_LOCK:
        if force_refresh or _FENCES_CACHE is None:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            rows = cur.execute("SELECT id,name,kind,geojson,color,updated_at FROM fences ORDER BY id").fetchall()
            con.close()
            keys = ["id", "name", "kind", "geojson", "color", "updated_at"]
            out = [dict(zip(keys, r)) for r in rows]
            for r in out:
                try:
                    r["geojson"] = json.loads(r["geojson"])
                except Exception:
                    pass
            _FENCES_CACHE = out
        return [dict(r) for r in (_FENCES_CACHE or [])]


def list_hss(force_refresh=False):
    global _HSS_CACHE
    with _SPATIAL_CACHE_LOCK:
        if force_refresh or _HSS_CACHE is None:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            rows = cur.execute("SELECT id,name,lat,lon,radius,active,updated_at FROM hss WHERE active=1 ORDER BY id").fetchall()
            con.close()
            keys = ["id", "name", "lat", "lon", "radius", "active", "updated_at"]
            _HSS_CACHE = [dict(zip(keys, r)) for r in rows]
        return [dict(r) for r in (_HSS_CACHE or [])]


def distance_m(lat1, lon1, lat2, lon2):
    """Iki enlem/boylam aras?ndaki mesafeyi metre cinsinden hesapla (haversine)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_inside_hss(lat, lon):
    """Aktif HSS'lerden herhangi birinin i�inde mi?"""
    try:
        for it in list_hss():
            h_lat = float(it["lat"])
            h_lon = float(it["lon"])
            r = float(it["radius"])
            d = distance_m(lat, lon, h_lat, h_lon)
            if d <= r:
                return True
    except Exception as e:
        print("?? is_inside_hss hata:", e)
    return False


def insert_hss(name, lat, lon, radius):
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("INSERT INTO hss(name,lat,lon,radius,active,updated_at) VALUES (?,?,?,?,1,?)",
                (name, float(lat), float(lon), float(radius), now_iso()))
    con.commit();
    _id = cur.lastrowid;
    con.close()
    list_hss(force_refresh=True)
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
    list_hss(force_refresh=True)


def delete_hss(hid):
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("DELETE FROM hss WHERE id=?", (int(hid),))
    con.commit();
    con.close()
    list_hss(force_refresh=True)


def save_kamikaze_row(payload: dict):
    """/api/kamikaze_bilgisi gelen paketi kal?c? kaydet"""
    if not SAVE_MATCH_DATA_TO_DB:
        return
    import json, datetime
    ts_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    kaynak = payload.get("kaynak_takim")  # yoksa None kal?r
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
    if DISABLE_HISTORY_QUERIES or not SAVE_MATCH_DATA_TO_DB:
        return []
    """UI tarih filtresi i�in socket?ten sorgulan?r"""
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
    # JSON alanlar? dict?e �evir
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
    if DISABLE_HISTORY_QUERIES or not SAVE_MATCH_DATA_TO_DB:
        return []
    """
    telemetry tablosundan filtreli veri �eker.
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

    # dict listeye �evir
    keys = ["ts_utc", "takim", "enlem", "boylam", "irtifa", "hiz", "batarya"]
    return [dict(zip(keys, r)) for r in rows]


def query_locks_history(kaynak=None, start_iso=None, end_iso=None, limit=1000):
    if DISABLE_HISTORY_QUERIES or not SAVE_MATCH_DATA_TO_DB:
        return []
    import sqlite3

    q = """
        SELECT ts_utc,
               kaynak_takim,
               kilitlenen_takim,
               otonom_kilitlenme,
               kilit_bitis_gps
        FROM locks
        WHERE 1 = 1 \
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
    if not SAVE_MATCH_DATA_TO_DB:
        return
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
    # print("? telemetry?DB tak?m=", takim)


def save_lock_row(team: int, payload: dict):
    if not SAVE_MATCH_DATA_TO_DB:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    kb = payload.get("kilitlenmeBitisZamani", {})

    kilit_bitis_json = json.dumps(kb)

    cur.execute("""
                INSERT INTO locks (ts_utc,
                                   kaynak_takim,
                                   kilitlenen_takim,
                                   otonom_kilitlenme,
                                   kilit_bitis_gps,
                                   extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    utc_now(),
                    team,  # kaynak_takim
                    payload.get("kilitlenen_takim", 0),  # varsa
                    int(payload.get("otonom_kilitlenme", 0)),
                    kilit_bitis_json,
                    json.dumps(payload)  # t�m payload saklan?r
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

    # Ger�ek login token?
    if tok in ISSUED_TOKENS:
        return True

    return False


# =========================
#  PUANLAMA (TEST - limitsiz)
# =========================

# team -> skor d�k�m�
SCORES = {}  # {team: {"auto_lock":0,"manual_lock":0,"kamikaze":0,"hss_sec":0,"tel_err_sec":0,"ihrac":0,"total":0.0}}
OFFLINE_AFTER_SEC = 2.5  # 2.5 sn telemetri yoksa offline say (rate-limit 0.5s oldu?u i�in makul)

# Anl?k durumlar
HSS_INSIDE = {}  # team -> bool
TEL_LAST_SEEN = {}  # team -> time.time()
TEL_LAST_HASH = {}  # team -> str
TEL_LAST_CHANGE = {}  # team -> time.time()
TEL_WARNED_AT = {}  # team -> time.time()   (ikaz edildi -> 10 sn say)
IHRAC_DONE = set()  # bir kez ihra� yaz (tekrar tekrar ceza yazmas?n)

HSS_WARNED = set()  # HSS i�inde olan tak?mlar (uyar? i�in)
GEOFENCE_OUTSIDE = {}  # team -> bool
GEOFENCE_WARNED_AT = {}  # team -> ts
GEOFENCE_IHRAC_DONE = set()
GEOFENCE_LAST = {}  # team -> last geofence check ts (watchdog i�in opsiyonel)

TEL_LAND_CALL = set()
TEL_LAND_REASONS = {}

TEAM_STATE = {}

TEL_STATE = {}
# team -> {
#   "mode": "ok" | "sabit" | "kesik" | "offline_paused",
#   "start": timestamp,
#   "ihrac_written": False
# }
FLIGHT_STATE = {}

# =========================
#  U�U? MODU KONTROL�
# =========================
# Sadece bu modlar cezas?z kabul edilir.
# Telemetri paketinde mode bilgisi varsa ?u alan adlar?ndan biri okunur:
# iha_modu, iha_mode, ucus_modu, u�u?_modu, flight_mode, mode
ALLOWED_FLIGHT_MODES = {"AUTO", "GUIDED", "FBWA"}
LAST_FLIGHT_MODE = {}          # team -> son g�r�len normalize mod
LAST_BAD_MODE_PENALIZED = {}   # team -> ceza yaz?lan son hatal? mod


def telemetry_signature(t):
    """
    Sabit telemetri kontrol� konum/h?z �zerinden yap?lmamal?.
    U�aklar ayn? konumda bekleyebilir; bu normaldir.

    Bu y�zden sadece paket i�indeki GPS saatinin ilerleyip ilerlemedi?ine bak?yoruz.
    gps_saati de?i?iyorsa telemetri canl?d?r, lat/lon/alt/h?z ayn? kalsa bile sabit say?lmaz.
    """
    if not isinstance(t, dict):
        t = {}

    gps = t.get("gps_saati") or t.get("gpsSaati") or t.get("gps_time") or {}

    def norm(v):
        try:
            return int(float(v))
        except Exception:
            return None

    if isinstance(gps, dict):
        sig_obj = {
            "saat": norm(gps.get("saat") or gps.get("hour")),
            "dakika": norm(gps.get("dakika") or gps.get("minute")),
            "saniye": norm(gps.get("saniye") or gps.get("second")),
            "milisaniye": norm(gps.get("milisaniye") or gps.get("millisecond") or gps.get("ms")),
        }
    else:
        # Baz? istemciler gps_saati'ni string/number olarak g�nderebilir.
        sig_obj = {"gps_saati": str(gps)}

    # gps_saati hi� gelmiyorsa eski konum-h?z hash'ine d�nme.
    # ��nk� eski davran??, ayn? konumda bekleyen u�a?? yanl??l?kla sabit say?yordu.
    # Bu durumda t�m bo? saatler ayn? imzay? �retir ve uyar? verir; do?ru ��z�m istemcinin gps_saati g�ndermesidir.
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
            # POZ?T?F
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

            # TEK SEFERL?K CEZA
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
    """Bearer token -> tak?m numaras? (fake_token_123 ise None d�ner)"""
    hdr = request.headers.get("Authorization", "")
    if not hdr.startswith("Bearer "):
        return None
    tok = hdr.split(" ", 1)[1].strip()
    # fake token testte var ama tak?m bilgisi yok
    if tok == "fake_token_123":
        return None
    return ISSUED_TOKENS.get(tok)


def is_team_online_for_scoring(team):
    """
    Puanlama i�in tak?m ger�ekten online m??
    Telemetri OFFLINE_AFTER_SEC s�resinden eskiyse puan/ceza i?lememeli.
    """
    try:
        team = int(team)
    except Exception:
        return False

    last_seen = TEL_LAST_SEEN.get(team, 0)
    return (time.time() - last_seen) <= OFFLINE_AFTER_SEC


def get_flight_mode_from_telemetry(t):
    """
    Telemetri paketinden u�u? modunu okur ve b�y�k harfe normalize eder.
    Alan yoksa None d�ner; b�ylece eski istemciler ceza yemez.
    Kabul edilen �rnek alanlar:
      iha_modu, iha_mode, ucus_modu, u�u?_modu, flight_mode, mode
    """
    if not isinstance(t, dict):
        return None

    for key in ("iha_modu", "iha_mode", "ucus_modu", "u�u?_modu", "flight_mode", "mode"):
        value = t.get(key)
        if value not in (None, ""):
            return str(value).strip().upper()

    return None


def check_flight_mode_penalty(team, t):
    """
    AUTO / GUIDED / FBWA d???ndaki moda ge�ilirse manual_limit cezas? yazar.
    Ayn? hatal? mod i�in her telemetride ceza yazmaz; sadece moda ge�i?te 1 kez yazar.
    Offline tak?mda veya puanlama kapal?yken ceza yaz?lmaz.
    """
    try:
        team = int(team)
    except Exception:
        return

    mode = get_flight_mode_from_telemetry(t)
    if mode is None:
        return

    previous_mode = LAST_FLIGHT_MODE.get(team)
    LAST_FLIGHT_MODE[team] = mode

    # ?zinli moda d�n�nce ayn? hatal? mod tekrar g�r�l�rse yeni ihlal say?labilsin.
    if mode in ALLOWED_FLIGHT_MODES:
        LAST_BAD_MODE_PENALIZED.pop(team, None)
        return

    if not SCORING_ACTIVE or not is_team_online_for_scoring(team):
        return

    # Ayn? hatal? modda kal?yorsa tekrar ceza yazma.
    if LAST_BAD_MODE_PENALIZED.get(team) == mode:
        return

    _ensure_team(team)
    SCORES[team]["manual_limit"] += 1
    LAST_BAD_MODE_PENALIZED[team] = mode
    _recalc_total(team)
    emit_score_update()
    print(f"?? MOD CEZASI: Tak?m {team} izin verilmeyen moda ge�ti: {mode} | �nceki={previous_mode}")


def emit_score_update():
    global _LAST_SCORE_EMIT_TS
    now_emit = time.time()
    if now_emit - _LAST_SCORE_EMIT_TS < SCORE_EMIT_MIN_PERIOD:
        return
    _LAST_SCORE_EMIT_TS = now_emit

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
            "hatali_kilit_kamikaze": s["hatali_kilit_kamikaze"],  # ? KR?T?K
            "manual_limit": s["manual_limit"],
            "hss_sec": s["hss_sec"],

            "area_violation": s["area_violation"],
            "boundary_violation": s["boundary_violation"],
            "ihrac": s["ihrac"],

            "total": s["total"],
            "online": is_team_online_for_scoring(team)
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

    # hedef alanlar?
    for f in ["hedef_merkez_X", "hedef_merkez_Y",
              "hedef_genislik", "hedef_yukseklik"]:
        if payload.get(f) in (None, "", 0):
            errors.append(f"eksik:{f}")

    return errors


def _recalc_total(team: int):
    _ensure_team(team)
    s = SCORES[team]

    total = 0.0

    # ? POZ?T?F PUANLAR
    total += s["auto_lock"] * 500
    total += s["manual_lock"] * 50
    total += s["kamikaze"] * 300
    total += s["auto_land"] * 100
    total += s["auto_takeoff"] * 50
    total += s["video_tx"] * 50

    # ? S�REL? CEZALAR
    total -= s["tel_err_sec"] * 0.2
    total -= s["hatali_kilit_kamikaze"] * 30
    total -= s["manual_limit"] * 10
    total -= s["hss_sec"] * 5

    # ? TEK SEFERL?K CEZALAR
    total -= s["area_violation"] * 150
    total -= s["boundary_violation"] * 200
    total -= s["ihrac"] * 75

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

    # kullan?c?y? bul
    user = next((u for u in VALID_USERS if u["kadi"] == kadi and u["sifre"] == sifre), None)
    if not user:
        return ("Ge�ersiz kullan?c? ad? veya ?ifre", 400)

    # rastgele token �ret
    token = secrets.token_hex(16)
    ISSUED_TOKENS[token] = user["takim"]

    print(f"? Login OK: {kadi} ? team {user['takim']} | token={token}")

    return jsonify({
        "takim_numarasi": user["takim"],
        "token": token
    }), 200


@app.route("/api/sunucusaati", methods=["GET"])
def sunucusaati():
    return jsonify(server_now_dict()), 200


# ?eman: tam olarak kullan?c?n?n g�nderdi?i alanlar
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

        # Aral?k kontrolleri (makul s?n?rlar)
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
        return "? HSS oturum a�ma hatas? 1", 401

    d = request.get_json(silent=True) or {}
    # enabled true/false bekliyoruz
    HSS_SEND_ENABLED = bool(d.get("enabled"))
    print("? HSS_SEND_ENABLED =", HSS_SEND_ENABLED)
    return jsonify({"ok": True, "enabled": HSS_SEND_ENABLED}), 200


@app.route("/api/hss_toggle", methods=["POST"])
def hss_toggle():
    """HSS sistemini aktif/pasif yap (u�ak taraf?nda ka�?nma aktif/pasif)"""
    global HSS_SYSTEM_ACTIVE, HSS_ACTIVE_SINCE
    if not ok_auth():
        return "? HSS oturum a�ma hatas? 2", 401

    d = request.get_json(silent=True) or {}
    previous_active = HSS_SYSTEM_ACTIVE

    # "active": true/false ile kontrol
    if "active" in d:
        HSS_SYSTEM_ACTIVE = bool(d["active"])
    else:
        # Toggle
        HSS_SYSTEM_ACTIVE = not HSS_SYSTEM_ACTIVE

    # HSS ilk kez aktif olduysa 60 sn cezas?z s�reyi ba?lat.
    # Zaten aktifken tekrar active=true gelirse s�reyi s?f?rlama.
    if HSS_SYSTEM_ACTIVE and not previous_active:
        HSS_ACTIVE_SINCE = time.time()
    elif not HSS_SYSTEM_ACTIVE:
        HSS_ACTIVE_SINCE = None

    status = "AKT?F ?" if HSS_SYSTEM_ACTIVE else "PAS?F ?"
    print(f"? HSS S?STEM?: {status}")

    # SocketIO ile t�m clientlara bildir
    socketio.emit("hss_system_status", {"hss_aktif": HSS_SYSTEM_ACTIVE})

    return jsonify({
        "ok": True,
        "hss_aktif": HSS_SYSTEM_ACTIVE,
        "message": f"HSS sistemi {status}"
    }), 200


@app.route("/api/telemetri_gonder", methods=["POST"])
def telemetri():
    # TEAM_NO art?k yay?n/ayr?m i�in kullan?lm?yor; projede ba?ka yerde kullan?yorsan?z kals?n.
    global _latest_telemetry, TEAM_NO, _TELEMETRY_STALE_SEC, _last_telemetry_ts, _RATE_PERIOD

    if not ok_auth():
        return "? telemetri oturum a�ma hatas?", 401

    t = request.get_json(silent=True) or {}
    # print("? Gelen Telemetri:", t)

    # ?ema/alan kontrol�: ba?ar?s?zsa 204 (g�vde yok)
    if not validate_telemetry(t):
        if LOG_TELEMETRY:
            print("validate_telemetry hata")
        return "", 204

    # 0) Tak?m id'yi G�NDERENDEN al
    try:
        takim = int(t["takim_numarasi"])

        # ? tak?m ilk kez ba?land?ysa skor tablosuna 0'la d�?s�n
        new_team = (takim not in SCORES)
        _ensure_team(takim)
        # ilk kez g�rd�ysek UI'a hemen yolla (0 puanla g�r�ns�n)
        if new_team:
            _recalc_total(takim)  # total = 0.0
            emit_score_update()
    except Exception:
        return ("bad request", 400)

    # 1) Rate limit (tak?m bazl?) ? 2 Hz (0.5 s)
    now_m = time.monotonic()
    last = _last_telemetry_ts.get(takim, 0)
    if now_m - last < _RATE_PERIOD:
        return jsonify({"ok": True, "ignored": "rate_limit"}), 200
    _last_telemetry_ts[takim] = now_m
    LAST_ACCEPTED_TELEMETRY_REAL_TS[takim] = time.time()

    # 2) In-memory son telemetri kayd? (tak?m bazl?)
    try:
        _latest_telemetry[takim] = {"telemetry": t, "ts": time.time()}
        OFFLINE_EMITTED.discard(takim)  # yeniden veri geldiyse art?k online say?labilir
    except Exception as e:
        print("? _latest_telemetry g�ncelleme hatas?:", e)

    # 3) (Opsiyonel) Geofence kontrol� ? g�nderene uygula
    try:
        lat = float(t.get("iha_enlem"))
        lon = float(t.get("iha_boylam"))

        GEOFENCE_LAST[takim] = time.time()  # ? opsiyonel: en son ne zaman kontrol edildi

        outside = not is_inside_fences(lat, lon)
        prev_outside = GEOFENCE_OUTSIDE.get(takim)
        GEOFENCE_OUTSIDE[takim] = outside

        # Socket yükünü azalt: sadece durum değişince yayınla.
        if prev_outside != outside:
            if outside:
                socketio.emit("geofence_violation", {
                    "takim": takim,
                    "lat": lat,
                    "lon": lon,
                    "utc": now_iso()
                })
            else:
                socketio.emit("geofence_ok", {"takim": takim, "utc": now_iso()})
    except Exception as e:
        print("?? Geofence kontrol hatas?:", e)

    # 3b) (Opsiyonel) HSS kontrol� ? g�nderene uygula
    try:
        lat = float(t.get("iha_enlem"))
        lon = float(t.get("iha_boylam"))

        inside = bool(HSS_SEND_ENABLED and HSS_SYSTEM_ACTIVE and is_inside_hss(lat, lon))

        prev_inside = HSS_INSIDE.get(takim, False)
        HSS_INSIDE[takim] = inside

        # ? HSS uyar?s?n? an?nda g�ster (cezadan ba??ms?z)
        # ?�erideyken 2Hz telemetriyle s�rekli "hss_inside" gelir ? UI watchdog d�?mez.
        # Socket yükünü azalt: HSS durumunu her telemetride değil, değişince yayınla.
        if inside and not prev_inside:
            socketio.emit("hss_inside", {"takim": takim, "utc": now_iso()})
        elif (not inside) and prev_inside:
            socketio.emit("hss_ok", {"takim": takim, "utc": now_iso()})
    except Exception as e:
        print("?? HSS kontrol hatas?:", e)
        HSS_INSIDE[takim] = False
        socketio.emit("hss_ok", {"takim": takim, "utc": now_iso()})

    # 4) DB kayıt kapalı: telemetri geçmişe/loga yazılmaz.
    # save_telemetry_row(takim, t) çağrısı performans için tamamen kaldırıldı.

    # 5) Enemies: t�m g�ncel telemetriler (kendisi dahil)
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
        print("enemies olu?tururken hata:", e)
        enemies = []

    # =========================
    # ? K?L?T TAK?P G�NCELLEME
    # =========================
    try:
        update_lock_tracking(takim, t, enemies)
    except Exception as e:
        print("?? update_lock_tracking hata:", e)

    # ---- PUAN: telemetri izleme state'i (emit'ten �nce) ----
    now = time.time()
    _ensure_team(takim)
    TEL_LAST_SEEN[takim] = now

    # =========================
    #  U�U? MODU KONTROL�
    # =========================
    check_flight_mode_penalty(takim, t)

    # ? TELEMETR? GER? GELD?YSE MODE RESET
    state = TEL_STATE.get(takim)
    if state and state.get("mode") in ("kesik", "offline_paused"):
        TEL_STATE[takim] = {
            "mode": "ok",
            "start": now,
            "ihrac_written": False
        }

        # Offline'dan d�ner d�nmez "sabit telemetri" cezas? yemesin
        TEL_LAST_CHANGE[takim] = now

    sig = telemetry_signature(t)

    if TEL_LAST_HASH.get(takim) != sig:
        TEL_LAST_HASH[takim] = sig
        TEL_LAST_CHANGE[takim] = now
    else:
        # GPS saati de?i?mediyse change time g�ncellenmez.
        # Konum/h?z ayn? diye art?k sabit saym?yoruz; karar sadece saate g�re veriliyor.
        if takim not in TEL_LAST_CHANGE:
            TEL_LAST_CHANGE[takim] = now

    # =========================
    #  OTONOM KALKI? / ?N?? TAK?B?
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

    # YERDE ? HAVADA
    if not state.get("airborne", False) and alt > 20:
        state["airborne"] = True
        if SCORING_ACTIVE and is_team_online_for_scoring(takim) and otonom == 1:
            print("? OTONOM KALKI?")
            SCORES[takim]["auto_takeoff"] += 1
            _recalc_total(takim)

    elif state.get("airborne", False) and alt < 5:
        state["airborne"] = False
        if SCORING_ACTIVE and is_team_online_for_scoring(takim) and otonom == 1:
            print("? OTONOM ?N??")
            SCORES[takim]["auto_land"] += 1
            _recalc_total(takim)

    FLIGHT_STATE[takim] = state

    # 6) skor g�ncelle
    try:
        _recalc_total(takim)
        emit_score_update()
    except Exception as _e:
        print("score update hata:", _e)

    # 7) HTTP cevab? ayn? formatta (UI geriye uyumlu)
    return jsonify({
        "sunucusaati": server_now_dict(),
        "konumBilgileri": enemies,
        "hss_koordinat_bilgileri": [
            {"id": it["id"], "hssEnlem": it["lat"], "hssBoylam": it["lon"], "hssYaricap": it["radius"]}
            for it in list_hss()
        ] if (HSS_SEND_ENABLED and HSS_SYSTEM_ACTIVE) else []
    }), 200


def _invalid_lock(team):
    _ensure_team(team)

    if SCORING_ACTIVE and is_team_online_for_scoring(team):
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

ACTIVE_LOCK_START = {}  # team -> lock_start_time
CURRENT_TARGET = {}  # team -> target_id
LAST_LOCKED_TARGET = {}  # team -> last_success_target
LAST_LOCK_SIG = {}  # duplicate lock kontrol


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

    # hedef kimli?i (basit bounding box imzas?)
    target_sig = (
        t.get("hedef_merkez_X"),
        t.get("hedef_merkez_Y"),
        t.get("hedef_genislik"),
        t.get("hedef_yukseklik")
    )

    # -------------------------
    # 1?? Kilit kapal?ysa reset
    # -------------------------
    if not kilit:
        ACTIVE_LOCK_START.pop(team, None)
        CURRENT_TARGET.pop(team, None)
        return

    # -------------------------
    # 2?? Hedef kaybolduysa reset
    # -------------------------
    if not hedef_var:
        ACTIVE_LOCK_START.pop(team, None)
        CURRENT_TARGET.pop(team, None)
        return

    # -------------------------
    # 3?? Yeni hedef ba?lad?ysa reset + ba?lat
    # -------------------------
    if CURRENT_TARGET.get(team) != target_sig:
        CURRENT_TARGET[team] = target_sig
        ACTIVE_LOCK_START[team] = time.time()
        return

    # -------------------------
    # 4?? Ayn? hedef devam ediyorsa s�re akmaya devam eder
    # -------------------------
    if team not in ACTIVE_LOCK_START:
        ACTIVE_LOCK_START[team] = time.time()


@app.route("/api/kilitlenme_bilgisi", methods=["POST"])
def kilitlenme():
    if not ok_auth():
        return "? kilitlenme oturum a�ma hatas? 1", 401

    data = request.get_json(silent=True) or {}

    # ? Tak?m sadece token'dan
    team = auth_team()
    if team is None:
        return "? kilitlenme oturum a�ma hatas? 2", 401

    team = int(team)
    _ensure_team(team)

    # =========================
    # 1?? ZORUNLU ALAN
    # =========================
    kb = data.get("kilitlenmeBitisZamani")
    if not isinstance(kb, dict):
        _invalid_lock(team)
        print("? kilitlenmeBitisZamani dict de?il veya eksik")
        return "", 204

    for f in ("saat", "dakika", "saniye", "milisaniye"):
        if f not in kb:
            _invalid_lock(team)
            print(f"? kilitlenmeBitisZamani i�inde '{f}' alan? eksik")
            return "", 204

    try:
        ok_flag = int(data.get("otonom_kilitlenme"))
    except Exception:
        _invalid_lock(team)
        print("? otonom_kilitlenme parse hatas?")
        return "", 204

    if ok_flag not in (0, 1):
        _invalid_lock(team)
        print("? otonom_kilitlenme ge�ersiz")
        return "", 204

    # =========================
    # 2?? TELEMETR? GER�EK K?L?T VAR MI?
    # =========================
    latest = _latest_telemetry.get(team, {}).get("telemetry", {})

    if latest.get("iha_kilitlenme") != 1:
        print("? Telemetride kilit aktif de?il")
        _invalid_lock(team)
        return "", 204

    # hedef bilgileri zorunlu
    if not all([
        latest.get("hedef_merkez_X"),
        latest.get("hedef_merkez_Y"),
        latest.get("hedef_genislik"),
        latest.get("hedef_yukseklik")
    ]):
        print("? Hedef X/Y/W/H eksik")
        _invalid_lock(team)
        return "", 204

    # video son 1 saniyede gelmi? mi
    # last_video = LAST_VIDEO_TS.get(team)
    # if not last_video or (time.time() - last_video) > 1.0:
    #    print(f"? Tak?m {team}: son 1 saniyede g�r�nt� yok, kilit ge�ersiz")
    #    _invalid_lock(team)
    #    return "", 204
    #
    # =========================
    # 3?? 4 SAN?YE TAK?P KONTROL�
    # =========================
    start_time = ACTIVE_LOCK_START.get(team)
    # if not start_time:
    #    print("? Lock start yok")
    #    _invalid_lock(team)
    #    return "", 204

    # duration = time.time() - start_time
    # if duration < 4:
    #    print("? 4 saniye dolmad?:", duration)
    #    _invalid_lock(team)
    #    return "", 204

    # =========================
    # 4?? HEDEF TEKRAR KONTROL�
    # =========================
    target = CURRENT_TARGET.get(team)
    # if target is None:
    #    print("? Target yok")
    #    _invalid_lock(team)
    #    return "", 204

    # if LAST_LOCKED_TARGET.get(team) == target:
    #    print("? Ayn? hedefe tekrar kilit")
    #    _invalid_lock(team)
    #    return "", 204

    # =========================
    # 5?? DUPLICATE ZAMAN
    # =========================
    lock_sig = (
        int(kb["saat"]),
        int(kb["dakika"]),
        int(kb["saniye"]),
        int(kb["milisaniye"])
    )

    if LAST_LOCK_SIG.get(team) == lock_sig:
        print("? Duplicate paket")
        _invalid_lock(team)
        return "", 204

    LAST_LOCK_SIG[team] = lock_sig

    # =========================
    # 6?? PUANLAMA
    # =========================
    if SCORING_ACTIVE and is_team_online_for_scoring(team):
        if ok_flag == 1:
            SCORES[team]["auto_lock"] += 1
        else:
            SCORES[team]["manual_lock"] += 1

        _recalc_total(team)
        emit_score_update()

    LAST_LOCKED_TARGET[team] = target

    # =========================
    # 7?? DB
    # =========================
    try:
        save_lock_row(team, data)
    except Exception as e:
        print("? save_lock_row:", e)

    # =========================
    # 8?? UI
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
        return "? kamikaze oturum a�ma hatas?", 401

    d = request.get_json(silent=True) or {}

    # ? TEAM MUTLAKA TOKEN'DAN
    team = auth_team()
    if team is None:
        team = d.get("kaynak_takim")

    if team is None:
        print("? TAKIM BULUNAMADI")
        return "", 204

    team = int(team)
    _ensure_team(team)
    d["kaynak_takim"] = team

    kb = d.get("kamikazeBaslangicZamani", {})
    ke = d.get("kamikazeBitisZamani", {})

    # ? HATALI PAKET
    if not (
            all(k in kb for k in ("saat", "dakika", "saniye", "milisaniye")) and
            all(k in ke for k in ("saat", "dakika", "saniye", "milisaniye")) and
            d.get("qrMetni")
    ):
        if SCORING_ACTIVE and is_team_online_for_scoring(team):
            SCORES[team]["hatali_kilit_kamikaze"] += 1
            _recalc_total(team)
            emit_score_update()

        print("? HATALI KAM?KAZE -30 yaz?ld?")
        return "", 204

    # ? DO?RU PAKET
    try:
        save_kamikaze_row(d)
        print("? kamikaze?DB OK")
    except Exception as e:
        print("? save_kamikaze_row HATA:", e)

    if SCORING_ACTIVE and is_team_online_for_scoring(team):
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

    print("? Kamikaze:", d)

    return "OK", 200


QR_COORD = {"qrEnlem": 38.70129013, "qrBoylam": 27.45397528, "qrWidth": 80.0, "qrHeight": 50.0}
@app.route("/api/qr_koordinati", methods=["GET"])
def qr_get():
    if not ok_auth():
        return "? QR oturum a�ma hatas?", 401
    return jsonify(QR_COORD), 200


@app.route("/api/qr_koordinati", methods=["POST"])
def qr_set():
    if not ok_auth():
        return "? QR oturum a�ma hatas?", 401

    data = request.get_json(silent=True) or {}
    changed = False

    if "qrEnlem" in data and "qrBoylam" in data:
        QR_COORD["qrEnlem"] = float(data["qrEnlem"])
        QR_COORD["qrBoylam"] = float(data["qrBoylam"])
        changed = True

    if "qrWidth" in data and "qrHeight" in data:
        QR_COORD["qrWidth"] = float(data["qrWidth"])
        QR_COORD["qrHeight"] = float(data["qrHeight"])
        changed = True

    if changed:
        if data.get("saveToFile"):
            try:
                import re
                with open(__file__, "r", encoding="utf-8") as f:
                    content = f.read()
                new_line = (
                    f'QR_COORD = {{"qrEnlem": {QR_COORD["qrEnlem"]}, '
                    f'"qrBoylam": {QR_COORD["qrBoylam"]}, '
                    f'"qrWidth": {QR_COORD["qrWidth"]}, '
                    f'"qrHeight": {QR_COORD["qrHeight"]}}}'
                )
                content = re.sub(
                    r'^QR_COORD\s*=\s*\{.*\}\s*$',
                    new_line, content, flags=re.MULTILINE
                )
                with open(__file__, "w", encoding="utf-8") as f:
                    f.write(content)

                # ui.html dosyas?n? da g�ncelle
                try:
                    import os
                    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
                    if os.path.exists(ui_path):
                        with open(ui_path, "r", encoding="utf-8") as f:
                            ui_content = f.read()
                        ui_new_line = f'window.qrSize = {{ w: {QR_COORD["qrWidth"]}, h: {QR_COORD["qrHeight"]} }};'
                        ui_content = re.sub(
                            r'window\.qrSize\s*=\s*\{.*?\};',
                            ui_new_line, ui_content
                        )
                        with open(ui_path, "w", encoding="utf-8") as f:
                            f.write(ui_content)
                except Exception as e:
                    print("ui.html guncellenemedi:", e)

            except Exception as e:
                print("Dosyalar guncellenemedi:", e)

        socketio.emit('qr_update', QR_COORD)
        return jsonify({"ok": True, "coord": QR_COORD}), 200

    return jsonify({"ok": False, "error": "Ge�ersiz veri"}), 400


# helper:
def ok_auth_or_dev():
    # URL parametresi ile dev mod a�?l?rsa yetkisiz eri?ime izin ver
    if request.args.get("dev") == "1":
        return True
    return ok_auth()


def ok_auth_or_public_hss():
    """
    HSS endpointi, yki_gorev_yazilimi'ndaki fetch_hss() ile
    (�o?u zaman Authorization/cookie olmadan) �a?r?labildi?i i�in
    auth YOKSA bile ge�elim. Auth varsa yine kabul.
    """
    return ok_auth() or True  # HSS'i public yap


@app.route("/api/hss", methods=["GET"])
def api_hss_list():
    return jsonify({"ok": True, "items": list_hss()}), 200


@app.route("/api/hss", methods=["POST"])
def api_hss_create():
    if not ok_auth(): return "? HSS oturum a�ma hatas? 3", 401
    d = request.get_json(silent=True) or {}
    name = (d.get("name") or "").strip() or "HSS"
    lat = d.get("lat")
    lon = d.get("lon")
    radius = d.get("radius")
    try:
        _id = insert_hss(name, float(lat), float(lon), float(radius))
    except Exception as e:
        return jsonify({
            "ok": False,
            "hata": "HSS olu?turulamad?",
            "sebep": str(e),  # �rn: "could not convert string to float: None"
            "eksik_alan": {
                "lat": lat,
                "lon": lon,
                "radius": radius
            }
        }), 400
    try:
        socketio.emit("hss_update", {"items": list_hss(force_refresh=True)})
    except:
        pass
    return jsonify({"ok": True, "id": _id}), 200


@app.route("/api/hss/<int:hid>", methods=["PUT"])
def api_hss_update(hid):
    if not ok_auth(): return "? HSS oturum a�ma hatas? 4", 401
    d = request.get_json(silent=True) or {}
    try:
        update_hss(hid, **d)
        socketio.emit("hss_update", {"items": list_hss(force_refresh=True)})
    except Exception as e:
        return jsonify({
            "ok": False,
            "hata": "HSS g�ncellenemedi",
            "sebep": str(e),
            "hss_id": hid
        }), 400
    return jsonify({"ok": True}), 200


@app.route("/api/hss/<int:hid>", methods=["DELETE"])
def api_hss_delete(hid):
    if not ok_auth(): return "? HSS oturum a�ma hatas? 5", 401
    delete_hss(hid)
    try:
        socketio.emit("hss_update", {"items": list_hss(force_refresh=True)})
    except:
        pass
    return jsonify({"ok": True}), 200


@app.route("/api/hss_koordinatlari", methods=["GET"])
def hss_public():
    # HSS koordinatlar? u�aklara sadece HSS g�nderimi A�IK ve sistem AKT?F ise payla??l?r.
    # HSS pasifken endpoint �al???r ama bo? liste d�ner.
    if not (HSS_SEND_ENABLED and HSS_SYSTEM_ACTIVE):
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
    if not ok_auth(): return "? HSS oturum a�ma hatas? 6", 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "Geofence"
    kind = data.get("kind")
    gj = data.get("geojson")
    color = data.get("color") or "#ef4444"
    if kind not in ("polygon",) or not gj:
        return jsonify({"ok": False, "error": "invalid payload 1"}), 400
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("INSERT INTO fences(name,kind,geojson,color,updated_at) VALUES(?,?,?,?,?)",
                (name, kind, json.dumps(gj), color, now_iso()))
    con.commit();
    con.close()
    items = list_fences(force_refresh=True)
    socketio.emit("fences_update", {"items": items})
    return jsonify({"ok": True, "items": items}), 200


@app.route("/api/fences/<int:fid>", methods=["PUT"])
def update_fence(fid):
    if not ok_auth(): return "? HSS oturum a�ma hatas? 7", 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or "Geofence"
    kind = data.get("kind")
    gj = data.get("geojson")
    color = data.get("color") or "#ef4444"
    if kind not in ("polygon",) or not gj:
        return jsonify({"ok": False, "error": "invalid payload 2"}), 400
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("UPDATE fences SET name=?,kind=?,geojson=?,color=?,updated_at=? WHERE id=?",
                (name, kind, json.dumps(gj), color, now_iso(), fid))
    con.commit();
    con.close()
    items = list_fences(force_refresh=True)
    socketio.emit("fences_update", {"items": items})
    return jsonify({"ok": True, "items": items}), 200


@app.route("/api/fences/<int:fid>", methods=["DELETE"])
def delete_fence(fid):
    if not ok_auth(): return "? HSS oturum a�ma hatas? 8", 401
    con = sqlite3.connect(DB_PATH);
    cur = con.cursor()
    cur.execute("DELETE FROM fences WHERE id=?", (fid,))
    con.commit();
    con.close()
    items = list_fences(force_refresh=True)
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
    global TEL_LAND_CALL, TEL_LAND_REASONS

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

    LAST_BAD_MODE_PENALIZED.clear()
    LAST_FLIGHT_MODE.clear()

    TEL_LAND_CALL = set()
    TEL_LAND_REASONS = {}

    emit_score_update()

    return jsonify({
        "ok": True,
        "message": "Skorlar s?f?rland?"
    }), 200


@app.route("/api/scoring_status", methods=["GET"])
def api_scoring_status():
    return jsonify({
        "ok": True,
        "scoring_active": SCORING_ACTIVE
    }), 200


def point_in_polygon(lat, lon, polygon_latlon):
    # polygon_latlon: [[lat,lon], ...] (ilk/son kapanmasa da i?ler)
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
            # ? OFFLINE ise: hi�bir ceza ilerlemesin HSS
            offline = (now - last_seen) > OFFLINE_AFTER_SEC
            if offline:
                # HSS / geofence durur
                GEOFENCE_OUTSIDE[team] = False
                GEOFENCE_WARNED_AT.pop(team, None)
                HSS_INSIDE[team] = False
                _recalc_total(team)
                continue

            if SCORING_ACTIVE:
                # HSS cezas?
                if HSS_SYSTEM_ACTIVE and HSS_INSIDE.get(team, False):
                    HSS_WARNED.add(team)

                    # HSS aktif edildikten sonraki ilk 60 saniyede ceza sayma.
                    hss_grace_done = (
                        HSS_ACTIVE_SINCE is not None and
                        (now - HSS_ACTIVE_SINCE) >= HSS_PENALTY_GRACE_SEC
                    )
                    if hss_grace_done:
                        SCORES[team]["hss_sec"] += 1

                        if SCORES[team]["hss_sec"] >= 30:
                            land_call(team, "HSS ihlali (30 sn)")
                else:
                    HSS_WARNED.discard(team)

                # Geofence cezas?
                if GEOFENCE_OUTSIDE.get(team, False):
                    if team not in GEOFENCE_WARNED_AT:
                        GEOFENCE_WARNED_AT[team] = now

                    if (now - GEOFENCE_WARNED_AT[team]) >= 10.0:
                        if team not in GEOFENCE_IHRAC_DONE:
                            SCORES[team]["boundary_violation"] += 1
                            GEOFENCE_IHRAC_DONE.add(team)
                            land_call(team, "S?n?r d??? ihlali (10 sn)")
                else:
                    GEOFENCE_WARNED_AT.pop(team, None)
                    GEOFENCE_IHRAC_DONE.discard(team)

                _recalc_total(team)
            else:
                # Puanlama kapal?ysa sadece uyar? state'leri temiz kals?n
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
            # GPS saati en az birka� saniye de?i?miyorsa sabit say.
            # 1 saniye �ok agresifti; saniye ��z�n�rl�kl� saatlerde yanl?? alarm �retir.
            sabit = (now - last_change) >= 3.0

            state = TEL_STATE.get(team, {
                "mode": "ok",
                "start": now,
                "ihrac_written": False
            })

            # 1) TELEMETR? OFFLINE
            if offline:
                # Offline olan tak?m?n puanlamas? tamamen durur.
                # Telemetri hatas? s�resi artmaz, ihra� yaz?lmaz, HSS/geofence ilerlemez.
                state = {
                    "mode": "offline_paused",
                    "start": now,
                    "ihrac_written": False
                }

                HSS_INSIDE[team] = False
                HSS_WARNED.discard(team)

                GEOFENCE_OUTSIDE[team] = False
                GEOFENCE_WARNED_AT.pop(team, None)
                GEOFENCE_IHRAC_DONE.discard(team)

                _recalc_total(team)

            # 2) TELEMETR? SAB?T
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
                        print(f"?HRA� (SAB?T): Tak?m {team}")
                        SCORES[team]["ihrac"] += 1
                        land_call(team, "Telemetri sabit (10 sn)")
                        state["ihrac_written"] = True
                        _recalc_total(team)

            # 3) HER ?EY NORMAL
            else:
                state = {
                    "mode": "ok",
                    "start": now,
                    "ihrac_written": False
                }

                if SCORING_ACTIVE:
                    SCORES[team]["tel_err_sec"] = 0

            TEL_STATE[team] = state

            # HSS/geofence süre cezalarını aynı watchdog döngüsünde hesapla.
            # Böylece score_tick_loop + telemetry_watchdog iki ayrı 1Hz loop olarak çalışmaz.
            if not offline:
                if SCORING_ACTIVE:
                    if HSS_SYSTEM_ACTIVE and HSS_INSIDE.get(team, False):
                        HSS_WARNED.add(team)
                        hss_grace_done = (
                            HSS_ACTIVE_SINCE is not None and
                            (now - HSS_ACTIVE_SINCE) >= HSS_PENALTY_GRACE_SEC
                        )
                        if hss_grace_done:
                            SCORES[team]["hss_sec"] += 1
                            if SCORES[team]["hss_sec"] >= 30:
                                land_call(team, "HSS ihlali (30 sn)")
                    else:
                        HSS_WARNED.discard(team)

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
                    HSS_WARNED.discard(team)

        emit_score_update()


def telemetry_broadcast_loop():
    while True:
        try:
            now = time.time()

            # 1 Hz alt? kontrol�
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

            # T�m aktif tak?mlar?n son telemetrisini 1 Hz yay?nla
            # Not: Eski kayd? yay?nlamaya devam ederse UI taraf? tak?m? hi� offline g�remez.
            for team, item in list(_latest_telemetry.items()):
                if now - item.get("ts", 0) > OFFLINE_AFTER_SEC:
                    if team not in OFFLINE_EMITTED:
                        socketio.emit("team_offline", {"takim": int(team)})
                        OFFLINE_EMITTED.add(team)
                    continue

                telem = item.get("telemetry", {})
                # Dashboard zaten her aktif takım için ayrı telemetry_update alıyor.
                # Burada enemies listesini tekrar üretmek O(n^2) yük oluşturuyordu.
                socketio.emit("telemetry_update", {
                    "takim": team,
                    "telemetry": telem,
                    "sunucusaati": server_now_dict(),
                    "enemies": []
                })

        except Exception as e:
            print("? telemetry_broadcast_loop hata:", e)

        socketio.sleep(1.0)


@socketio.on('connect')
def on_connect():
    pass


@socketio.on('disconnect')
def on_disconnect():
    pass


from flask_socketio import emit
from flask import request


@socketio.on('fetch_history')
def on_fetch_history(payload):
    # Telemetri geçmişi/logları kapalı. Yeni kayıt tutulmaz, History DB okunmaz.
    emit('history_result', {"ok": True, "rows": [], "count": 0}, room=request.sid)


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
        print("? fetch_locks hata:", e)
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
        print("? fetch_kamikaze hata:", e)
        emit('kamikaze_result', {"ok": False, "error": str(e)}, room=request.sid)


if __name__ == "__main__":
    print("DB PATH =", os.path.abspath("iha_logs.db"))
    init_db()
    # Performans: score_tick_loop içindeki HSS/geofence süre takibi telemetry_watchdog içine alındı.
    # Ayrı bir 1Hz skor loop'u başlatmıyoruz.
    socketio.start_background_task(telemetry_watchdog)
    socketio.start_background_task(telemetry_broadcast_loop)
    socketio.run(app, host="0.0.0.0", port=10001, debug=False, use_reloader=False)