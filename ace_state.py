import sqlite3, os, time, hmac, hashlib
DB = os.environ.get("ACE_STATE_DB", "ace_workspace/deploy_state.db")
SECRET = os.environ.get("ACE_DEPLOY_SECRET", "dev-secret-change-me")
def _conn():
    con = sqlite3.connect(DB, timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    return con
def _sign(key, value):
    return hmac.new(SECRET.encode(), f"{key}:{value}".encode(), hashlib.sha256).hexdigest()
def set_flag(key, value, by="manual"):
    s = _sign(key, value)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO control_flags (id,key,value,updated_at,updated_by,signature) "
                  "VALUES ((SELECT id FROM control_flags WHERE key=?), ?, ?, ?, ?, ?)",
                  (key, key, value, ts, by, s))
    return True
def get_flag(key):
    with _conn() as c:
        r = c.execute("SELECT value,signature FROM control_flags WHERE key=?", (key,)).fetchone()
    if not r:
        return None
    value, sig = r
    if _sign(key, value) != sig:
        raise RuntimeError("control flag signature mismatch")
    return value
