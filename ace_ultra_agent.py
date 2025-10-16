#!/usr/bin/env python3
"""
🚀 ACE ULTRA AGENT v7 (Deployer debounce + ENABLE_DEPLOY)
✨ 10 Agent | 9 AI | Tam Otomasyon
⚙️ Deployer artık Config.ENABLE_DEPLOY ile kontrol edilir ve debounce içerir.
"""
from __future__ import annotations
import os
import re
import sys
import time
import json
import queue
import sqlite3
import threading
import subprocess
import requests
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# ============= OPTIONAL IMPORTS =============
try:
    import tkinter as tk
    from tkinter import scrolledtext
    HAS_TK = True
except Exception:
    HAS_TK = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import openpyxl
    HAS_EXCEL = True
except Exception:
    HAS_EXCEL = False

try:
    from reportlab.pdfgen import canvas
    HAS_PDF = True
except Exception:
    HAS_PDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

try:
    import replicate
    HAS_REPLICATE = True
except Exception:
    HAS_REPLICATE = False

try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

try:
    import elevenlabs
    HAS_ELEVENLABS = True
except Exception:
    HAS_ELEVENLABS = False

try:
    import undetected_chromedriver as uc
    HAS_UNDETECTED_CHROMEDRIVER = True
except Exception:
    HAS_UNDETECTED_CHROMEDRIVER = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except Exception:
    HAS_PLAYWRIGHT = False

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

try:
    from diffusers import StableDiffusionPipeline
    import torch
    HAS_DIFFUSERS = True
except Exception:
    HAS_DIFFUSERS = False

try:
    import audiocraft
    from audiocraft.models import MusicGen
    HAS_AUDIOCRAFT = True
except Exception:
    HAS_AUDIOCRAFT = False

# ============= CONFIG ============
class Config:
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_HEALTH: str = os.environ.get("OLLAMA_HEALTH", "http://localhost:11434/health")
    MODELS: Dict[str, str] = {
        "qwen": "qwen2.5-coder:latest",
        "llama": "llama3.1:8b",
        "mistral": "mistral-nemo:latest",
    }
    WORKSPACE_DIR: str = os.environ.get("ACE_WS", "ace_workspace")
    ENABLE_SECURITY: bool = True
    HTTP_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.5
    QUOTA_DB: str = "quota.db"
    SECRETS_FILE: str = "secrets.json"

    # Yeni: deploy kontrolü ve debounce süresi (saniye)
    ENABLE_DEPLOY: bool = False   # False ise Deployer hiç çalışmaz; test sonrası True yap
    DEPLOY_COOLDOWN: int = 10    # Deployer için minimum saniye aralığı

# ============= UTILS ============
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
    full = os.path.join(Config.WORKSPACE_DIR, relpath)
    full = os.path.abspath(full)
    base = os.path.abspath(Config.WORKSPACE_DIR)
    if not full.startswith(base):
        raise PermissionError("Sandbox violation")
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
        cls._conn.execute(
            "CREATE TABLE IF NOT EXISTS quota(id INTEGER PRIMARY KEY, filename TEXT, bytes INTEGER, created TIMESTAMP)"
        )
        cls._conn.commit()
    @classmethod
    def record(cls, filename: str, bytes_: int) -> None:
        if cls._conn is None:
            cls.init()
        with cls._lock:
            cls._conn.execute(
                "INSERT INTO quota(filename, bytes, created) VALUES (?, ?, ?)",
                (filename, int(bytes_), datetime.utcnow()),
            )
            cls._conn.commit()

Quota.init()

# ============= HTTP UTILS ============
def http_post_with_retry(url: str, json_payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> requests.Response:
    backoff = Config.BACKOFF_FACTOR
    for attempt in range(Config.MAX_RETRIES):
        try:
            r = requests.post(url, json=json_payload, headers=headers or {}, timeout=Config.HTTP_TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == Config.MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= Config.BACKOFF_FACTOR

def http_get_with_retry(url: str, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    backoff = Config.BACKOFF_FACTOR
    for attempt in range(Config.MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers or {}, timeout=Config.HTTP_TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == Config.MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= Config.BACKOFF_FACTOR

# ============= OLLAMA ============
def check_ollama_health() -> bool:
    try:
        r = requests.get(Config.OLLAMA_HEALTH, timeout=Config.HTTP_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False

def call_ollama(prompt: str, model_key: str) -> str:
    if not check_ollama_health():
        raise RuntimeError("Ollama unavailable or unhealthy")
    model = Config.MODELS.get(model_key, model_key)
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = http_post_with_retry(Config.OLLAMA_URL, payload)
    try:
        data = r.json()
        return data.get("response", json.dumps(data))
    except Exception:
        return r.text

# ============= ML SYSTEM ============
class MLSystem:
    def __init__(self) -> None:
        self.ai_perf: Dict[str, Dict[str, float]] = {
            "code": {"qwen": 0.9, "llama": 0.85, "mistral": 0.8},
            "web": {"qwen": 0.85, "llama": 0.9, "mistral": 0.8},
            "doc": {"llama": 0.9, "qwen": 0.85},
        }

    def get_best(self, cat: str) -> str:
        if cat not in self.ai_perf:
            cat = "code"
        perf = self.ai_perf[cat]
        return max(perf, key=perf.get)

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

# ============= BASE AGENT ============
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

# ============= AGENTS ============
class RequirementsAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            low = req.lower()
            if any(k in low for k in ["scrape", "kopyala", "web sitesi", "web"]):
                typ = "web"
            elif "api" in low or "flask" in low:
                typ = "api"
            elif any(k in low for k in ["resim", "image", "görüntü"]):
                typ = "image"
            elif any(k in low for k in ["video", "vidyo"]):
                typ = "video"
            elif any(k in low for k in ["müzik", "music", "ses", "audio"]):
                typ = "music"
            elif any(k in low for k in ["oyun", "game"]):
                typ = "game"
            elif any(k in low for k in ["email", "e-posta", "e posta"]):
                typ = "email"
            elif any(k in low for k in ["bilet", "flight", "satın al", "satınal"]):
                typ = "automation"
            else:
                typ = "app"
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
            prompt = f"Generate compact, well documented, safe Python code for: {design}"
            try:
                code = call_ollama(prompt, model)
            except Exception:
                code = "# Fallback code - model unreachable\nprint('fallback')\n"
            self.send("ReviewerAgent", code)

class ReviewerAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            code = msg.content
            ok = any(k in code for k in ["def ", "import ", "print("])
            feedback = "" if ok else "Basit static kontrol başarısız"
            result = {"approved": ok, "feedback": feedback, "code": code}
            self.send("TesterAgent", result)

class TesterAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            data = msg.content
            code = data["code"]
            if "os.system" in code or "subprocess" in code:
                result = {"passed": False, "logs": "Riskli sistem çağrısı tespit edildi", "code": code}
            else:
                result = {"passed": True, "logs": "", "code": code}
            self.send("DebuggerAgent", result)

class DebuggerAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            data = msg.content
            if not data["passed"] or not data.get("approved", True):
                code = data["code"]
                error = data.get("logs") or data.get("feedback") or "Bilinmeyen hata"
                fixed_code = f"# Otomatik düzeltme: {error}\n" + code
                self.send("DocumenterAgent", fixed_code)
            else:
                self.send("DocumenterAgent", data["code"])

class DocumenterAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            code = msg.content
            header = "# Auto-generated documentation\n"
            doc = header + code
            self.send("IntegratorAgent", doc)

class IntegratorAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            code = msg.content
            final = code
            safe_write("main.py", final)
            self.notify("Kod oluşturuldu: main.py")
            self.send("DeployerAgent", final)

class DeployerAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, gui: Optional["GUI"] = None):
        super().__init__(name, bus, gui)
        self._last_deploy = 0.0
        self._deploy_lock = threading.Lock()

    def handle_message(self, msg: Message):
        if msg.recipient != self.name:
            return
        # Eğer global olarak deploy kapalıysa atla
        if not Config.ENABLE_DEPLOY:
            self.notify("Deploy devre dışı (Config.ENABLE_DEPLOY=False).")
            return
        # Debounce kontrolü
        now = time.time()
        with self._deploy_lock:
            if now - self._last_deploy < max(0.0, Config.DEPLOY_COOLDOWN):
                remaining = int(max(0, Config.DEPLOY_COOLDOWN - (now - self._last_deploy)))
                self.notify(f"Önceki deploy yakın zamanda başladı, atlandı ({remaining}s bekleyin).")
                return
            self._last_deploy = now

        code = msg.content
        try:
            dockerfile = "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\nRUN pip install -r requirements.txt --no-input\nCMD [\"python\",\"main.py\"]\n"
            safe_write("Dockerfile", dockerfile)
            safe_write("requirements.txt", "flask\nrequests\nbeautifulsoup4\n")
            compose_text = "version: '3.8'\nservices:\n  ace_agent:\n    build: .\n    ports:\n      - \"8080:8080\"\n    volumes:\n      - ./"+ Config.WORKSPACE_DIR +":/app/"+ Config.WORKSPACE_DIR +"\n    environment:\n      - OLLAMA_URL="+ Config.OLLAMA_URL +"\n"
            safe_write("docker-compose.yml", compose_text)
            self.notify("Deploy başlatıldı (yerel deneme).")
        except Exception as e:
            self.notify(f"Deploy hatası: {e}")

class ImageAIAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            if "stable diffusion" in req.lower():
                self._generate_stable_diffusion(req)
            elif "replicate" in req.lower():
                self._generate_replicate(req)
            elif "leonardo" in req.lower():
                self._generate_leonardo(req)
            else:
                self._generate_stable_diffusion(req)

    def _generate_stable_diffusion(self, req: str):
        if not HAS_DIFFUSERS:
            self.notify("Stable Diffusion yüklü değil")
            return
        try:
            prompt = req.replace("stable diffusion", "").strip()
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            image = pipe(prompt).images[0]
            path = os.path.join(Config.WORKSPACE_DIR, "sd_image.png")
            image.save(path)
            Quota.record("sd_image.png", os.path.getsize(path))
            self.notify("Stable Diffusion resmi oluşturuldu: sd_image.png")
        except Exception as e:
            self.notify(f"Stable Diffusion hatası: {e}")

    def _generate_replicate(self, req: str):
        if not HAS_REPLICATE:
            self.notify("Replicate yüklü değil")
            return
        try:
            secrets = load_secrets()
            token = secrets.get("replicate", {}).get("api_token")
            if not token:
                self.notify("Replicate token bulunamadı")
                return
            replicate.Client(api_token=token)
            prompt = req.replace("replicate", "").strip()
            output = replicate.run(
                "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                input={"prompt": prompt}
            )
            url = output[0] if isinstance(output, list) else output
            r = requests.get(url)
            path = os.path.join(Config.WORKSPACE_DIR, "replicate_image.png")
            with open(path, "wb") as f:
                f.write(r.content)
            Quota.record("replicate_image.png", os.path.getsize(path))
            self.notify("Replicate resmi oluşturuldu: replicate_image.png")
        except Exception as e:
            self.notify(f"Replicate hatası: {e}")

    def _generate_leonardo(self, req: str):
        try:
            secrets = load_secrets()
            api_key = secrets.get("leonardo", {}).get("api_key")
            if not api_key:
                self.notify("Leonardo API key bulunamadı")
                return
            prompt = req.replace("leonardo", "").strip()
            headers = {
                "Authorization": f"Bearer {api_key}"}
            payload = {
                "prompt": prompt,
                "width": 512,
                "height": 512
            }
            r = http_post_with_retry("https://cloud.leonardo.ai/api/rest/v1/generations", payload, headers)
            data = r.json()
            generation_id = data["sdGenerationJob"]["generationId"]
            for _ in range(30):
                r = http_get_with_retry(f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}", headers)
                data = r.json()
                if data["generations_by_pk"]["status"] == "COMPLETE":
                    url = data["generations_by_pk"]["generated_images"][0]["url"]
                    img_r = requests.get(url)
                    path = os.path.join(Config.WORKSPACE_DIR, "leonardo_image.png")
                    with open(path, "wb") as f:
                        f.write(img_r.content)
                    Quota.record("leonardo_image.png", os.path.getsize(path))
                    self.notify("Leonardo resmi oluşturuldu: leonardo_image.png")
                    return
                time.sleep(2)
            self.notify("Leonardo resmi oluşturulamadı (timeout)")
        except Exception as e:
            self.notify(f"Leonardo hatası: {e}")

class VideoAIAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            if "capcut" in req.lower():
                self._generate_capcut(req)
            elif "runway" in req.lower():
                self._generate_runway(req)
            elif "d-id" in req.lower():
                self._generate_did(req)
            else:
                self._generate_runway(req)

    def _generate_capcut(self, req: str):
        self.notify("CapCut otomasyonu henüz tamamlanmadı")

    def _generate_runway(self, req: str):
        try:
            secrets = load_secrets()
            api_key = secrets.get("runway", {}).get("api_key")
            if not api_key:
                self.notify("Runway API key bulunamadı")
                return
            prompt = req.replace("runway", "").strip()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "style": "cinematic"
            }
            r = http_post_with_retry("https://api.runwayml.com/v1/video/generate", payload, headers)
            data = r.json()
            video_url = data.get("video_url")
            if video_url:
                video_r = requests.get(video_url)
                path = os.path.join(Config.WORKSPACE_DIR, "runway_video.mp4")
                with open(path, "wb") as f:
                    f.write(video_r.content)
                Quota.record("runway_video.mp4", os.path.getsize(path))
                self.notify("Runway videosu oluşturuldu: runway_video.mp4")
            else:
                self.notify("Runway videosu oluşturulamadı")
        except Exception as e:
            self.notify(f"Runway hatası: {e}")

    def _generate_did(self, req: str):
        try:
            secrets = load_secrets()
            api_key = secrets.get("did", {}).get("api_key")
            if not api_key:
                self.notify("D-ID API key bulunamadı")
                return
            prompt = req.replace("d-id", "").strip()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "script": {
                    "type": "text",
                    "input": prompt
                },
                "avatar_url": "https://d-id-public-bucket.s3.amazonaws.com/default_avatar.png",
                "config": {
                    "result_format": "mp4"
                }
            }
            r = http_post_with_retry("https://api.d-id.com/talks", payload, headers)
            data = r.json()
            talk_id = data["id"]
            for _ in range(30):
                r = http_get_with_retry(f"https://api.d-id.com/talks/{talk_id}", headers)
                data = r.json()
                if data["status"] == "done":
                    video_url = data["result_url"]
                    video_r = requests.get(video_url)
                    path = os.path.join(Config.WORKSPACE_DIR, "did_avatar.mp4")
                    with open(path, "wb") as f:
                        f.write(video_r.content)
                    Quota.record("did_avatar.mp4", os.path.getsize(path))
                    self.notify("D-ID avatar videosu oluşturuldu: did_avatar.mp4")
                    return
                time.sleep(2)
            self.notify("D-ID videosu oluşturulamadı (timeout)")
        except Exception as e:
            self.notify(f"D-ID hatası: {e}")

class MusicAIAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            if "suno" in req.lower():
                self._generate_suno(req)
            elif "elevenlabs" in req.lower():
                self._generate_elevenlabs(req)
            elif "audiocraft" in req.lower():
                self._generate_audiocraft(req)
            else:
                self._generate_elevenlabs(req)

    def _generate_suno(self, req: str):
        self.notify("Suno otomasyonu henüz tamamlanmadı")

    def _generate_elevenlabs(self, req: str):
        if not HAS_ELEVENLABS:
            self.notify("ElevenLabs yüklü değil")
            return
        try:
            secrets = load_secrets()
            api_key = secrets.get("elevenlabs", {}).get("api_key")
            if not api_key:
                self.notify("ElevenLabs API key bulunamadı")
                return
            elevenlabs.set_api_key(api_key)
            text = req.replace("elevenlabs", "").strip()
            audio = elevenlabs.generate(text=text)
            path = os.path.join(Config.WORKSPACE_DIR, "elevenlabs_audio.mp3")
            elevenlabs.save(audio, path)
            Quota.record("elevenlabs_audio.mp3", os.path.getsize(path))
            self.notify("ElevenLabs sesi oluşturuldu: elevenlabs_audio.mp3")
        except Exception as e:
            self.notify(f"ElevenLabs hatası: {e}")

    def _generate_audiocraft(self, req: str):
        if not HAS_AUDIOCRAFT:
            self.notify("Audiocraft yüklü değil")
            return
        try:
            text = req.replace("audiocraft", "").strip()
            model = MusicGen.get_pretrained('facebook/musicgen-small')
            model.set_generation_params(duration=8)
            wav = model.generate([text])
            path = os.path.join(Config.WORKSPACE_DIR, "audiocraft_music.wav")
            import torchaudio
            torchaudio.save(path, wav[0].cpu(), sample_rate=model.sample_rate)
            Quota.record("audiocraft_music.wav", os.path.getsize(path))
            self.notify("Audiocraft müzik oluşturuldu: audiocraft_music.wav")
        except Exception as e:
            self.notify(f"Audiocraft hatası: {e}")

class WebAutomationAgent(BaseAgent):
    def handle_message(self, msg: Message):
        if msg.recipient == self.name:
            req = msg.content
            if "bilet" in req.lower() or "flight" in req.lower():
                self._search_flights(req)
            elif "email" in req.lower():
                self._send_email(req)
            else:
                self._search_flights(req)

    def _search_flights(self, req: str):
        self.notify("Bilet arama otomasyonu henüz tamamlanmadı")

    def _send_email(self, req: str):
        try:
            m_to = re.search(r"to:([^\s]+)", req)
            m_subject = re.search(r"subject:([^\\n]+)", req)
            m_body = re.search(r"body:(.+)", req)
            to = m_to.group(1) if m_to else "user@example.com"
            subject = m_subject.group(1).strip() if m_subject else "ACE Agent Mesajı"
            body = m_body.group(1).strip() if m_body else "ACE Agent tarafından gönderildi."
            self._send_email_via_smtp(subject, body, to)
            self.notify(f"E-posta gönderildi: {to}")
        except Exception as e:
            self.notify(f"E-posta hatası: {e}\nSMTP yapılandırma rehberini okuyun.")

    def _send_email_via_smtp(self, subject: str, body: str, to: str) -> None:
        secrets = load_secrets()
        smtp_cfg = secrets.get("smtp")
        if not smtp_cfg:
            raise RuntimeError("SMTP yapılandırması yok. ace_workspace/secrets.json içine smtp bilgilerini ekleyin.")
        import smtplib
        from email.mime.text import MIMEText
        host = smtp_cfg.get("host")
        port = int(smtp_cfg.get("port", 587))
        username = smtp_cfg.get("username")
        password = smtp_cfg.get("password")
        use_tls = bool(smtp_cfg.get("use_tls", True))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = to
        server = smtplib.SMTP(host, port, timeout=Config.HTTP_TIMEOUT)
        try:
            if use_tls:
                server.starttls()
            server.login(username, password)
            server.sendmail(username, [to], msg.as_string())
        finally:
            server.quit()

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
            ImageAIAgent("ImageAIAgent", self.bus, self.gui),
            VideoAIAgent("VideoAIAgent", self.bus, self.gui),
            MusicAIAgent("MusicAIAgent", self.bus, self.gui),
            WebAutomationAgent("WebAutomationAgent", self.bus, self.gui),
        ]
        for agent in self.agents:
            agent.start()

    def process(self, user_req: str) -> None:
        try:
            low = user_req.lower()
            if any(k in low for k in ["resim", "image", "görüntü"]):
                self.bus.send(Message(sender="User", recipient="ImageAIAgent", content=user_req))
            elif any(k in low for k in ["video", "vidyo"]):
                self.bus.send(Message(sender="User", recipient="VideoAIAgent", content=user_req))
            elif any(k in low for k in ["müzik", "music", "ses", "audio"]):
                self.bus.send(Message(sender="User", recipient="MusicAIAgent", content=user_req))
            elif any(k in low for k in ["bilet", "flight", "satın al", "satınal", "email", "e-posta", "e posta"]):
                self.bus.send(Message(sender="User", recipient="WebAutomationAgent", content=user_req))
            else:
                self.bus.send(Message(sender="User", recipient="RequirementsAgent", content=user_req))
        except Exception as exc:
            self._notify(f"Hata: {exc}")

    def _notify(self, msg: str) -> None:
        if self.gui:
            self.gui.add_message(msg)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============= GUI ============
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ACE ULTRA AGENT v7")
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
        if Config.ENABLE_SECURITY and not SecurityLayer.sanitize_command(msg):
            self.add_message("❌ Tehlikeli komut engellendi!")
            return
        threading.Thread(target=self.pm.process, args=(msg,), daemon=True).start()

    def run(self):
        self.add_message("🚀 ACE ULTRA AGENT v7")
        self.add_message("✨ 10 Agent | 9 AI | Tam Otomasyon")
        self.add_message("💡 Örnekler:")
        self.add_message("")
        self.add_message('"kedi resmi üret"')
        self.add_message('"flyer hazırla"')
        self.add_message('"tetris yap"')
        self.add_message('"new york istanbul bilet ara"')
        self.add_message("")
        self.add_message("⚙️ Setup: secrets.json doldur")
        self.root.mainloop()

# ============= CLI ============
def run_cli_loop():
    pm = ProjectManager(None)
    print("ACE ULTRA AGENT v7 - Komut satırı modu. Çıkmak için 'exit' yazın.")
    print("💡 Örnekler:")
    print('"kedi resmi üret"')
    print('"müzik oluştur"')
    print('"bilet ara"')
    print("")
    while True:
        try:
            req = input("> ").strip()
            if not req:
                continue
            if req.lower() in ("exit", "quit"):
                break
            if Config.ENABLE_SECURITY and not SecurityLayer.sanitize_command(req):
                print("Tehlikeli komut engellendi!")
                continue
            threading.Thread(target=pm.process, args=(req,), daemon=True).start()
        except KeyboardInterrupt:
            break

# ============= DEPENDENCIES ============
DEPENDENCIES = [
    "requests", "beautifulsoup4", "pillow", "openpyxl", "reportlab",
    "diffusers", "transformers", "torch", "replicate", "groq",
    "elevenlabs", "selenium", "undetected-chromedriver", "playwright",
    "pygame", "audiocraft", "lxml", "accelerate"
]

def write_requirements():
    safe_write("requirements.txt", "\n".join(DEPENDENCIES) + "\n")

def write_readme_smtp():
    text = (
        "SMTP Rehberi\n\n"
        "1) ace_workspace/secrets.json dosyası oluşturun ve içine aşağıdaki yapıyı koyun:\n\n"
        "{\n"
        '  "smtp": {\n'
        '    "host": "smtp.example.com",\n'
        '    "port": 587,\n'
        '    "username": "you@example.com",\n'
        '    "password": "app-or-password",\n'
        '    "use_tls": true\n'
        '  }\n'
        "}\n\n"
        "2) Gmail kullanıyorsanız uygulama şifresi veya OAuth kullanın.\n"
        "3) SMTP bilgilerini girdikten sonra 'email to:someone@example.com subject:Test body:Merhaba' komutunu kullanın.\n"
    )
    safe_write("SMTP_README.txt", text)

def write_secrets_template():
    template = {
        "groq": {"api_key": ""},
        "replicate": {"api_token": ""},
        "leonardo": {"api_key": ""},
        "runway": {"api_key": "", "api_secret": ""},
        "did": {"api_key": ""},
        "canva": {"access_token": "", "refresh_token": ""},
        "elevenlabs": {"api_key": ""},
        "capcut": {"email": "", "password": ""},
        "suno": {"email": "", "password": ""},
        "smtp": {"host": "smtp.gmail.com", "port": 587, "username": "", "password": "", "use_tls": True}
    }
    path = os.path.join(Config.WORKSPACE_DIR, Config.SECRETS_FILE)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

# ============= SECURITY ============
class SecurityLayer:
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\bdel\s+",
        r"\bwget\b",
        r"\bcurl\b",
        r"\bdd\s+if=\b",
    ]

    @staticmethod
    def sanitize_command(cmd: str) -> bool:
        s = cmd.lower()
        for pat in SecurityLayer.DANGEROUS_PATTERNS:
            if re.search(pat, s):
                return False
        return True

    @staticmethod
    def sandbox_path(path: str) -> str:
        ensure_workspace()
        base = os.path.abspath(Config.WORKSPACE_DIR)
        full = os.path.abspath(path)
        if not full.startswith(base):
            raise PermissionError("Sandbox violation: path outside workspace")
        return full

# ============= MAIN ============
def main():
    parser = argparse.ArgumentParser(prog="ace_ultra_agent", description="ACE Ultra Agent v7")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui" if HAS_TK else "cli", help="Çalışma modu")
    args = parser.parse_args()
    ensure_workspace()
    write_requirements()
    write_readme_smtp()
    write_secrets_template()
    if args.mode == "gui" and HAS_TK:
        gui = GUI()
        gui.run()
    else:
        run_cli_loop()

if __name__ == "__main__":
    main()
