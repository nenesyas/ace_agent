"""
ACE ULTRA AGENT + Safe Deploy System (single file) - COMPLETE FIXED VERSION
Run: python ace_ultra_agent_deploy.py
"""

import os
import re
import sys
import time
import json
import queue
import sqlite3
import threading
import subprocess
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Optional libs
try:
    from filelock import FileLock
    HAS_FILELOCK = True
except Exception:
    HAS_FILELOCK = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except Exception:
    HAS_WATCHDOG = False
    class FileSystemEventHandler:
        pass
    class Observer:
        def __init__(self): pass
        def schedule(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def join(self, *args): pass

try:
    import tkinter as tk
    from tkinter import scrolledtext
    HAS_TK = True
except Exception:
    HAS_TK = False

try:
    import requests
except ImportError:
    requests = None

# ============= CONFIG ============
class Config:
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_HEALTH: str = "http://localhost:11434/health"
    MODELS: Dict[str, str] = {"qwen": "qwen2.5-coder:latest", "llama": "llama3.1:8b", "mistral": "mistral-nemo:latest"}
    WORKSPACE_DIR: str = "ace_workspace"
    ENABLE_SECURITY: bool = True
    HTTP_TIMEOUT: int = 10
    QUOTA_DB: str = "quota.db"
    SECRETS_FILE: str = "secrets.json"
    DEPLOY_COOLDOWN: int = 10

# ============= UTIL ============
def ensure_workspace() -> None:
    os.makedirs(Config.WORKSPACE_DIR, exist_ok=True)

def load_secrets() -> Dict[str, Any]:
    ensure_workspace()
    p = os.path.join(Config.WORKSPACE_DIR, Config.SECRETS_FILE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_write(relpath: str, content: str) -> str:
    ensure_workspace()
    full = os.path.abspath(os.path.join(Config.WORKSPACE_DIR, relpath))
    base = os.path.abspath(Config.WORKSPACE_DIR)
    if not full.startswith(base):
        raise PermissionError("Sandbox violation")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return full

# ============= QUOTA ============
class Quota:
    _conn: Optional[sqlite3.Connection] = None
    _lock = threading.Lock()
    
    @classmethod
    def init(cls) -> None:
        ensure_workspace()
        db = os.path.join(Config.WORKSPACE_DIR, Config.QUOTA_DB)
        cls._conn = sqlite3.connect(db, check_same_thread=False)
        cls._conn.execute("CREATE TABLE IF NOT EXISTS quota(id INTEGER PRIMARY KEY, filename TEXT, bytes INTEGER, created TIMESTAMP)")
        cls._conn.commit()
    
    @classmethod
    def record(cls, filename: str, bytes_: int) -> None:
        if cls._conn is None:
            cls.init()
        with cls._lock:
            cls._conn.execute("INSERT INTO quota(filename, bytes, created) VALUES (?, ?, ?)", (filename, int(bytes_), datetime.utcnow()))
            cls._conn.commit()

Quota.init()

# ============= MESSAGE BUS ============
@dataclass
class Message:
    sender: str
    recipient: str
    content: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class MessageBus:
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()
    
    def register_agent(self, agent_name: str):
        with self.lock:
            if agent_name not in self.queues:
                self.queues[agent_name] = queue.Queue()
    
    def send(self, msg: Message):
        with self.lock:
            if msg.recipient in self.queues:
                self.queues[msg.recipient].put(msg)
            else:
                for q in self.queues.values():
                    q.put(msg)
    
    def receive(self, agent_name: str, timeout: float = 1.0) -> Optional[Message]:
        try:
            return self.queues[agent_name].get(timeout=timeout)
        except queue.Empty:
            return None

class BaseAgent:
    def __init__(self, name: str, bus: MessageBus, gui: Optional["GUI"] = None):
        self.name = name
        self.bus = bus
        self.gui = gui
        self.bus.register_agent(name)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = True
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def run(self):
        while self.running:
            msg = self.bus.receive(self.name, timeout=0.5)
            if msg:
                try:
                    self.handle_message(msg)
                except Exception as e:
                    self.notify(f"Handler error: {e}")
    
    def handle_message(self, msg: Message):
        pass
    
    def send(self, recipient: str, content: Any):
        self.bus.send(Message(sender=self.name, recipient=recipient, content=content))
    
    def notify(self, msg: str):
        if self.gui:
            self.gui.add_message(f"[{self.name}] {msg}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{self.name}] {msg}")

# ============= ML SYSTEM ============
def check_ollama_health() -> bool:
    if not requests:
        return False
    try:
        r = requests.get(Config.OLLAMA_HEALTH, timeout=Config.HTTP_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False

def call_ollama(prompt: str, model_key: str) -> str:
    if not requests:
        return "# Requests not available\nprint('fallback')\n"
    if not check_ollama_health():
        raise RuntimeError("Ollama unavailable")
    payload = {"model": model_key, "prompt": prompt, "stream": False}
    r = requests.post(Config.OLLAMA_URL, json=payload, timeout=Config.HTTP_TIMEOUT)
    try:
        data = r.json()
        return data.get("response", json.dumps(data))
    except Exception:
        return r.text

class MLSystem:
    def __init__(self) -> None:
        self.ai_perf: Dict[str, Dict[str, float]] = {"code": {"qwen": 0.9, "llama": 0.85, "mistral": 0.8}}
    
    def get_best(self, cat: str) -> str:
        if cat not in self.ai_perf:
            cat = "code"
        perf = self.ai_perf[cat]
        return max(perf, key=perf.get)

# ============= AGENTS ============
class RequirementsAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            low = req.lower()
            typ = "api" if "api" in low or "flask" in low else "app"
            response = {"purpose": req, "type": typ}
            self.send("DesignerAgent", response)

class DesignerAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            spec = msg.content
            structure = ["main.py", "utils.py"] if spec.get("type") == "app" else ["api.py"]
            design = {"structure": structure, "tech": "python", "spec": spec}
            self.send("CoderAgent", design)

class CoderAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, gui: Optional["GUI"] = None):
        super().__init__(name, bus, gui)
        self.ml = MLSystem()
    
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            design = msg.content
            model = self.ml.get_best("code")
            prompt = f"Generate compact Python code for: {design}"
            try:
                code = call_ollama(prompt, model)
            except Exception:
                code = "# fallback\nprint('fallback')\n"
            self.send("ReviewerAgent", code)

class ReviewerAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            code = msg.content
            ok = any(k in code for k in ["def ", "import ", "print("])
            feedback = "" if ok else "Static check failed"
            self.send("TesterAgent", {"approved": ok, "feedback": feedback, "code": code})

class TesterAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            data = msg.content
            code = data["code"]
            passed = "os.system" not in code and "subprocess" not in code
            result = {"passed": passed, "logs": "" if passed else "Risky call", "code": code}
            self.send("DebuggerAgent", result)

class DebuggerAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            data = msg.content
            if not data["passed"] or not data.get("approved", True):
                code = data["code"]
                error = data.get("logs") or data.get("feedback") or "unknown"
                fixed_code = f"# Auto-fix: {error}\n" + code
                self.send("DocumenterAgent", fixed_code)
            else:
                self.send("DocumenterAgent", data["code"])

class DocumenterAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            code = msg.content
            doc = "# Auto-generated\n" + code
            self.send("IntegratorAgent", doc)

# ============= DEPLOY SYSTEM ============
WORKSPACE = Path(Config.WORKSPACE_DIR)
LOCK_FILE = WORKSPACE / ".deploy.lock"
STATE_DB = WORKSPACE / "deploy_state.db"
ENABLE_FILE = WORKSPACE / "deploy.enable"
CONTROL_LOG = WORKSPACE / "deploy_control.log"
OUTPUT_FILE = WORKSPACE / "integrated.html"
WATCH_DIRS = [WORKSPACE / "designs", WORKSPACE / "components", WORKSPACE / "assets"]
DOCKER_CONTAINER = "bloom_website"
DOCKER_TARGET_PATH = "/usr/share/nginx/html/index.html"

deploy_queue: List[Path] = []
queue_lock = threading.Lock()

def init_deploy_db():
    WORKSPACE.mkdir(exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS deploy_history (id INTEGER PRIMARY KEY, timestamp TEXT, status TEXT, message TEXT, build_log TEXT)")
    conn.commit()
    conn.close()

def log_deploy_db(status: str, message: str = "", build_log: str = ""):
    try:
        conn = sqlite3.connect(STATE_DB)
        conn.execute("INSERT INTO deploy_history (timestamp,status,message,build_log) VALUES (?,?,?,?)", (datetime.now().isoformat(), status, message, build_log))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DEPLOY DB] Error: {e}")

def log_control_action(action: str, user: str = "system"):
    timestamp = datetime.now().isoformat()
    entry = f"[{timestamp}] {user}: {action}\n"
    WORKSPACE.mkdir(exist_ok=True)
    with open(CONTROL_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[CONTROL] {entry.strip()}")

def is_deploy_enabled() -> bool:
    if not ENABLE_FILE.exists():
        return False
    content = ENABLE_FILE.read_text(encoding="utf-8").strip().lower()
    return content == "auto"

def enable_deploy(mode: str = "manual", user: str = "system"):
    if mode not in ["auto", "manual"]:
        raise ValueError("mode must be 'auto' or 'manual'")
    WORKSPACE.mkdir(exist_ok=True)
    ENABLE_FILE.write_text(mode, encoding="utf-8")
    log_control_action(f"Deploy ENABLED (mode: {mode})", user)

def disable_deploy(user: str = "system"):
    if ENABLE_FILE.exists():
        ENABLE_FILE.unlink()
    log_control_action("Deploy DISABLED", user)

def get_deploy_history(limit: int = 10) -> List[tuple]:
    try:
        conn = sqlite3.connect(STATE_DB)
        cur = conn.execute("SELECT timestamp, status, message FROM deploy_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []

def add_to_queue(html_file: Path):
    with queue_lock:
        if html_file not in deploy_queue:
            deploy_queue.append(html_file)
            print(f"[QUEUE] Added {html_file.name} (size={len(deploy_queue)})")

def deploy_to_docker(html_file: Path):
    if not is_deploy_enabled():
        print("[DEPLOY] BLOCKED - auto-deploy disabled")
        log_deploy_db("BLOCKED", "Auto-deploy disabled")
        return
    
    # Docker kontrolü
    try:
        docker_check = subprocess.run(["docker", "ps"], capture_output=True, timeout=5)
        docker_available = docker_check.returncode == 0
    except Exception:
        docker_available = False
    
    if not docker_available:
        print("[DEPLOY] ⚠ Docker not available - using local file copy")
        try:
            # Yerel deploy klasörüne kopyala
            local_deploy = WORKSPACE / "deployed"
            local_deploy.mkdir(exist_ok=True)
            target = local_deploy / "index.html"
            target.write_text(html_file.read_text(encoding="utf-8"), encoding="utf-8")
            log_deploy_db("SUCCESS", f"Local deploy: {html_file.name}")
            print(f"[DEPLOY] ✓ Local Success: {target}")
            return
        except Exception as e:
            log_deploy_db("ERROR", f"Local deploy failed: {e}")
            print(f"[DEPLOY] ✗ Local Error: {e}")
            return
    
    # Docker deploy
    try:
        print(f"[DEPLOY] → Processing: {html_file.name}")
        res = subprocess.run(["docker", "cp", str(html_file), f"{DOCKER_CONTAINER}:{DOCKER_TARGET_PATH}"], capture_output=True, text=True, timeout=20)
        if res.returncode != 0:
            raise Exception(res.stderr.strip() or "docker cp failed")
        reload = subprocess.run(["docker", "exec", DOCKER_CONTAINER, "nginx", "-s", "reload"], capture_output=True, text=True, timeout=10)
        if reload.returncode != 0:
            raise Exception(reload.stderr.strip() or "nginx reload failed")
        log_deploy_db("SUCCESS", f"Deployed {html_file.name}", res.stdout)
        print(f"[DEPLOY] ✓ Success: {html_file.name}")
    except subprocess.TimeoutExpired:
        log_deploy_db("ERROR", "timeout")
        print("[DEPLOY] ✗ Timeout")
    except Exception as e:
        log_deploy_db("ERROR", str(e))
        print(f"[DEPLOY] ✗ Error: {e}")

def process_queue_loop():
    while True:
        html_file = None
        with queue_lock:
            if deploy_queue:
                html_file = deploy_queue.pop(0)
        if not html_file:
            time.sleep(1)
            continue
        if HAS_FILELOCK:
            lock = FileLock(str(LOCK_FILE), timeout=30)
            try:
                with lock:
                    deploy_to_docker(html_file)
            except Exception as e:
                log_deploy_db("ERROR", f"Lock error: {e}")
                print(f"[QUEUE] Lock error: {e}")
        else:
            deploy_to_docker(html_file)

# ============= INTEGRATOR ============
class IntegratorHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_change = 0.0
        self.debounce_time = 2.0
    
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".html"):
            return
        now = time.time()
        if now - self.last_change < self.debounce_time:
            return
        self.last_change = now
        p = Path(event.src_path)
        print(f"[INTEGRATOR] Change detected: {p.name}")
        html_files = []
        for d in WATCH_DIRS:
            if d.exists():
                html_files.extend(list(d.glob("*.html")))
        if not html_files:
            print("[INTEGRATOR] No HTML files found")
            return
        latest = max(html_files, key=lambda f: f.stat().st_mtime)
        OUTPUT_FILE.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[INTEGRATOR] Integrated: {latest.name} -> {OUTPUT_FILE.name}")
        add_to_queue(OUTPUT_FILE)

def start_integrator_observer():
    if not HAS_WATCHDOG:
        print("[INTEGRATOR] watchdog not available")
        return None
    for d in WATCH_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    handler = IntegratorHandler()
    observer = Observer()
    for d in WATCH_DIRS:
        observer.schedule(handler, str(d), recursive=False)
    observer.start()
    return observer

# ============= INTEGRATOR & DEPLOYER AGENTS ============
class IntegratorAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, gui: Optional["GUI"] = None):
        super().__init__(name, bus, gui)
        self._save_lock = threading.Lock()
        self._debounce_seconds = 5.0
        self._deploy_flag_path = os.path.join(Config.WORKSPACE_DIR, "deploy.enable")
    
    def _deploy_allowed(self) -> bool:
        try:
            return os.path.exists(self._deploy_flag_path) and open(self._deploy_flag_path, "r", encoding="utf-8").read().strip().lower() == "auto"
        except Exception:
            return False
    
    def handle_message(self, msg: Message):
        if msg.recipient != self.name:
            return
        try:
            code = msg.content
            h = str(hash(code))
            now = time.time()
            with self._save_lock:
                last_saved = state_get("last_saved_hash")
                last_save_time = float(state_get("last_save_time") or "0")
                if last_saved == h:
                    self.notify("Integrator: içerik aynı, dosya tekrar yazılmadı.")
                    os.utime(os.path.join(Config.WORKSPACE_DIR, "main.py"), None)
                    state_set("last_save_time", str(now))
                    return
                if now - last_save_time < self._debounce_seconds and last_saved == h:
                    self.notify("Integrator atlandı: kısa tekrar.")
                    return
                safe_write("main.py", code)
                state_set("last_saved_hash", h)
                state_set("last_save_time", str(now))
                self.notify("Kod oluşturuldu: main.py")
                if self._deploy_allowed():
                    last_deployed = state_get("last_deploy_hash")
                    if last_deployed != h:
                        self.send("DeployerAgent", code)
                    else:
                        self.notify("Deployer atlandı: aynı kod daha önce deploy edilmiş.")
                else:
                    self.notify("Deployer çağrısı atlandı (deploy.enable yok veya auto değil).")
        except Exception as e:
            self.notify(f"Integrator error: {e}")

class DeployerAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, gui: Optional["GUI"] = None):
        super().__init__(name, bus, gui)
        self._last_deploy = 0.0
        self._deploy_lock = threading.Lock()
    
    def handle_message(self, msg: Message):
        if msg.recipient != self.name:
            return
        now = time.time()
        with self._deploy_lock:
            if now - self._last_deploy < Config.DEPLOY_COOLDOWN:
                remaining = int(Config.DEPLOY_COOLDOWN - (now - self._last_deploy))
                self.notify(f"Önceki deploy yakın, atlandı ({remaining}s).")
                return
            self._last_deploy = now
        code = msg.content
        try:
            dockerfile = "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\nCMD [\"python\",\"main.py\"]\n"
            safe_write("Dockerfile", dockerfile)
            safe_write("requirements.txt", "flask\nrequests\n")
            self.notify("Deploy başlatıldı (yerel deneme).")
            h = str(hash(code))
            state_set("last_deploy_hash", h)
            integrated = WORKSPACE / "integrated.html"
            if integrated.exists():
                add_to_queue(integrated)
        except Exception as e:
            self.notify(f"Deploy hatası: {e}")

# ============= STATE HELPERS ============
def init_state_table():
    ensure_workspace()
    db = os.path.join(Config.WORKSPACE_DIR, Config.QUOTA_DB)
    conn = sqlite3.connect(db, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()

def state_get(key: str) -> Optional[str]:
    db = os.path.join(Config.WORKSPACE_DIR, Config.QUOTA_DB)
    conn = sqlite3.connect(db, check_same_thread=False)
    cur = conn.execute("SELECT value FROM state WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def state_set(key: str, value: str) -> None:
    db = os.path.join(Config.WORKSPACE_DIR, Config.QUOTA_DB)
    conn = sqlite3.connect(db, check_same_thread=False)
    conn.execute("REPLACE INTO state(key,value) VALUES (?,?)", (key, value))
    conn.commit()
    conn.close()

init_state_table()
init_deploy_db()

# ============= PROJECT MANAGER ============
class ProjectManager:
    def __init__(self, gui: Optional["GUI"] = None) -> None:
        self.gui = gui
        self.bus = MessageBus()
        self.agents: List[BaseAgent] = []
        self._init_agents()
    
    def _init_agents(self):
        self.agents = [
            RequirementsAgent("RequirementsAgent", self.bus, self.gui),
            DesignerAgent("DesignerAgent", self.bus, self.gui),
            CoderAgent("CoderAgent", self.bus, self.gui),
            ReviewerAgent("ReviewerAgent", self.bus, self.gui),
            TesterAgent("TesterAgent", self.bus, self.gui),
            DebuggerAgent("DebuggerAgent", self.bus, self.gui),
            DocumenterAgent("DocumenterAgent", self.bus, self.gui),
            IntegratorAgent("IntegratorAgent", self.bus, self.gui),
            DeployerAgent("DeployerAgent", self.bus, self.gui),
        ]
        for agent in self.agents:
            agent.start()
    
    def process(self, user_req: str) -> None:
        self.bus.send(Message(sender="User", recipient="RequirementsAgent", content=user_req))

# ============= GUI ============
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ACE ULTRA AGENT")
        self.root.geometry("900x600")
        self.setup_ui()
        self.pm = ProjectManager(self)
    
    def setup_ui(self):
        self.chat = scrolledtext.ScrolledText(self.root, state=tk.DISABLED, bg="#0d001a", fg="#00ff00")
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.X, padx=8, pady=4)
        self.input = tk.Entry(frame, bg="#2a0050", fg="white", font=("Arial", 12))
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.input.bind("<Return>", lambda e: self.submit())
        btn = tk.Button(frame, text="GÖNDER", command=self.submit, bg="#ff00ff", fg="white")
        btn.pack(side=tk.RIGHT)
    
    def add_message(self, msg: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)
    
    def submit(self):
        msg = self.input.get().strip()
        if not msg:
            return
        self.add_message(f"SİZ: {msg}")
        self.input.delete(0, tk.END)
        if Config.ENABLE_SECURITY and any(re.search(p, msg.lower()) for p in [r"rm\s+-rf", r"wget", r"curl"]):
            self.add_message("Tehlikeli komut engellendi!")
            return
        threading.Thread(target=self.pm.process, args=(msg,), daemon=True).start()
    
    def run(self):
        self.add_message("ACE ULTRA AGENT")
        self.root.mainloop()

# ============= SYSTEM START ============
def print_status():
    status = {"enabled": is_deploy_enabled(), "mode": (ENABLE_FILE.read_text().strip() if ENABLE_FILE.exists() else None)}
    print("\n[STATUS]")
    print(f"  • Deploy Enabled: {status['enabled']}")
    print(f"  • Mode: {status['mode']}")
    print(f"  • Queue Size: {len(deploy_queue)}")
    history = get_deploy_history(5)
    if history:
        print("\n[HISTORY]")
        for ts, st, msg in history:
            print(f"  [{ts}] {st}: {msg}")

def start_background_services():
    t = threading.Thread(target=process_queue_loop, daemon=True)
    t.start()
    obs = start_integrator_observer()
    return obs

def main_interactive():
    pm = ProjectManager(None)
    obs = start_background_services()
    print("Starting agents. Type 'exit' to quit.")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            if line in ("exit", "quit"):
                break
            if line.startswith("enable"):
                mode = line.split()[1] if len(line.split()) > 1 else "manual"
                enable_deploy(mode, "interactive")
                print("Enabled")
            elif line.startswith("disable"):
                disable_deploy("interactive")
                print("Disabled")
            elif line.startswith("status"):
                print_status()
            elif line.startswith("history"):
                hist = get_deploy_history(20)
                for ts, st, msg in hist:
                    print(f"[{ts}] {st}: {msg}")
            elif line.startswith("deploy "):
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and os.path.exists(parts[1]):
                    add_to_queue(Path(parts[1]))
                    print("Added to queue")
                else:
                    print("File not found")
            else:
                pm.process(line)
    except KeyboardInterrupt:
        pass
    finally:
        if obs:
            obs.stop()
            obs.join()

def handle_commands():
    parser = argparse.ArgumentParser(prog="ace_ultra_agent_deploy")
    parser.add_argument("cmd", nargs="?", help="command")
    parser.add_argument("arg", nargs="?", help="argument")
    args = parser.parse_args()
    ensure_workspace()
    init_deploy_db()
    if not args.cmd:
        main_interactive()
        return
    cmd = args.cmd.lower()
    if cmd == "enable":
        mode = args.arg or "manual"
        enable_deploy(mode, "cli")
        print(f"Deploy enabled (mode={mode})")
    elif cmd == "disable":
        disable_deploy("cli")
        print("Deploy disabled")
    elif cmd == "status":
        print_status()
    elif cmd == "history":
        limit = int(args.arg) if args.arg else 20
        hist = get_deploy_history(limit)
        for ts, st, msg in hist:
            print(f"[{ts}] {st}: {msg}")
    elif cmd == "deploy":
        if not args.arg:
            print("Usage: deploy <file.html>")
            sys.exit(1)
        f = Path(args.arg)
        if not f.exists():
            print("File not found")
            sys.exit(1)
        enable_deploy("auto", "manual_cli")
        add_to_queue(f)
        print("Added to queue")
        # time.sleep(10)
        # disable_deploy("manual_cli")
        # print("Auto-disabled")
    else:
        print("Unknown command")
        sys.exit(1)

if __name__ == "__main__":
    handle_commands()
def ace_docker_deploy_integrated():
    from ace_docker import ensure_running, put_file, reload_nginx, backup_path, restore_tar
    import os, time

    container_name = "bloom_website"  # Eğer container adı farklıysa burayı değiştirin
    c = ensure_running(container_name)

    # optional: backup mevcut site içeriği
    backup_dir = "ace_workspace/backups"
    os.makedirs(backup_dir, exist_ok=True)
    bakfile = os.path.join(backup_dir, f"pre_deploy_{int(time.time())}.tar")
    try:
        backup_path(c, "/usr/share/nginx/html", bakfile)
    except Exception:
        pass

    # integrated.html'i yerleştir
    put_file(c, "ace_workspace/integrated.html", "/usr/share/nginx/html/index.html")
    try:
        from ace_docker_perms import ensure_path_perms
        ensure_path_perms(c, "/usr/share/nginx/html/index.html")
    except Exception:
        pass

    # nginx reload
    reload_nginx(c)



