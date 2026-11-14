"""
Module AI Google Studio (Gemini) đọc và thuyết minh lỗi vi phạm giao thông bằng tiếng Việt.
Đầu ra luôn tuân theo đúng định dạng chuẩn:
    "xe '<loại xe>' biển số <biển số> <lỗi của xe đó>"
Hỗ trợ:
  - Chuẩn hóa câu đọc tự nhiên qua Gemini API (Google AI Studio key).
  - Tạo giọng đọc AI tiếng Việt qua Gemini TTS (chuyển PCM -> WAV phát trực tiếp bằng winsound).
  - Fallback ngoại tuyến an toàn nếu mất mạng / quá hạn mức API.
  - Worker chạy trên QThread nền để không gây đơ hoặc giật kích thước giao diện PyQt6.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import os
import queue
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from PyQt6.QtCore import QThread, pyqtSignal

from core.database import ViolationRecord
from core.logger import get_app_logger

logger = get_app_logger("ai_narrator")

DEFAULT_GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY", "").strip()
AI_SETTINGS_PATH = "configs/ai_settings.json"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

VEHICLE_VI_MAP: Dict[str, str] = {
    "Car": "ô tô",
    "car": "ô tô",
    "Motorbike": "máy",
    "motorbike": "máy",
    "Motorcycle": "máy",
    "motorcycle": "máy",
    "Bus": "buýt",
    "bus": "buýt",
    "Truck": "tải",
    "truck": "tải",
}

VIOLATION_VI_RULES: Tuple[Tuple[str, str], ...] = (
    ("P.106a", "đi vào đường cấm xe tải P.106a"),
    ("Cam xe tai", "đi vào đường cấm xe tải"),
    ("P.102", "đi vào đường cấm P.102"),
    ("Di nguoc chieu", "đi ngược chiều"),
    ("Qua toc do", "chạy quá tốc độ quy định"),
    ("Vuot toc do", "chạy quá tốc độ quy định"),
    ("Dung do", "dừng đỗ trái phép"),
    ("Vuot den do", "vượt đèn đỏ"),
)

VEHICLE_EN_MAP: Dict[str, str] = {
    "Car": "car",
    "car": "car",
    "Motorbike": "motorbike",
    "motorbike": "motorbike",
    "Motorcycle": "motorbike",
    "motorcycle": "motorbike",
    "Bus": "bus",
    "bus": "bus",
    "Truck": "truck",
    "truck": "truck",
}

VIOLATION_EN_RULES: Tuple[Tuple[str, str], ...] = (
    ("P.106a", "entered no-truck zone P.106a"),
    ("Cam xe tai", "entered no-truck zone"),
    ("P.102", "entered restricted road P.102"),
    ("Di nguoc chieu", "drove the wrong way"),
    ("Qua toc do", "exceeded the speed limit"),
    ("Vuot toc do", "exceeded the speed limit"),
    ("Dung do", "parked illegally"),
    ("Vuot den do", "ran a red light"),
)


@dataclass
class AISettings:
    google_ai_api_key: str = DEFAULT_GOOGLE_AI_API_KEY
    text_model: str = "models/gemini-3.5-flash"
    tts_model: str = "models/gemini-2.5-flash-preview-tts"
    tts_voice: str = "Aoede"
    voice_enabled: bool = True
    use_ai_text_polish: bool = True
    use_ai_tts_audio: bool = True
    cooldown_seconds: float = 8.0
    ui_language: str = "vi"
    voice_language: str = "vi"
    speech_rate: float = 1.0


def load_ai_settings(path: str = AI_SETTINGS_PATH) -> AISettings:
    """Đọc cấu hình AI Google Studio từ JSON (ưu tiên GEMINI_API_KEY nếu có trong env)."""
    settings = AISettings()
    file_path = Path(path)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for key, val in raw.items():
                    if hasattr(settings, key):
                        setattr(settings, key, val)
        except Exception as exc:
            logger.warning("Không đọc được %s (%s), dùng mặc định.", path, exc)
    else:
        save_ai_settings(settings, path)

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        settings.google_ai_api_key = env_key
    elif Path(".env").exists():
        try:
            for line in Path(".env").read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        settings.google_ai_api_key = val
                        break
        except Exception:
            pass
    if not settings.google_ai_api_key:
        settings.google_ai_api_key = DEFAULT_GOOGLE_AI_API_KEY
    return settings


def save_ai_settings(settings: AISettings, path: str = AI_SETTINGS_PATH) -> None:
    """Lưu cấu hình AI Google Studio xuống JSON."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=2, ensure_ascii=False)


def translate_vehicle_type(vehicle_type: str, lang: str = "vi") -> str:
    clean = (vehicle_type or "").strip()
    if lang == "en":
        return VEHICLE_EN_MAP.get(clean, clean.lower() or "vehicle")
    mapped = VEHICLE_VI_MAP.get(clean, clean.lower() or "phương tiện")
    if mapped.startswith("xe "):
        mapped = mapped[3:].strip()
    return mapped


def format_plate_for_reading(plate_number: str, lang: str = "vi") -> str:
    clean = (plate_number or "").strip()
    if not clean or clean.upper() in {"KHÔNG ĐỌC ĐƯỢC", "KHONG DOC DUOC", "UNKNOWN", "UNREADABLE"}:
        return "unreadable" if lang == "en" else "không đọc được"
    return clean


def translate_violation_phrase(record: ViolationRecord, lang: str = "vi") -> str:
    vtype = (record.violation_type or "").strip()
    lower = vtype.lower()

    if lang == "en":
        matched = ""
        for token, phrase in VIOLATION_EN_RULES:
            if token.lower() in lower:
                matched = phrase
                break
        if not matched:
            matched = f"committed violation {vtype}" if vtype else "committed a traffic violation"
        if "speed" in matched and record.speed_kmh > 0:
            if record.is_speed_calibrated:
                matched = f"{matched} ({record.speed_kmh:.0f} km/h)"
            else:
                matched = f"{matched} (estimated {record.speed_kmh:.0f} km/h)"
        return matched

    matched = ""
    for token, phrase in VIOLATION_VI_RULES:
        if token.lower() in lower:
            matched = phrase
            break

    if not matched:
        matched = f"vi phạm {vtype}" if vtype else "vi phạm giao thông"

    if "quá tốc độ" in matched and record.speed_kmh > 0:
        if record.is_speed_calibrated:
            matched = f"{matched} ({record.speed_kmh:.0f} km/h)"
        else:
            matched = f"{matched} (ước tính {record.speed_kmh:.0f} km/h)"
    return matched


def format_violation_offline(record: ViolationRecord, voice_language: str = "vi") -> str:
    """
    Tạo câu đọc chuẩn xác theo ngôn ngữ:
      - Tiếng Việt: xe '<loại xe>' biển số <biển số> <lỗi của xe đó>
      - Tiếng Anh : vehicle <type> license plate <plate> <violation>
    """
    lang = "en" if str(voice_language).lower() == "en" else "vi"
    veh = translate_vehicle_type(record.vehicle_type, lang)
    plate = format_plate_for_reading(record.plate_number, lang)
    err = translate_violation_phrase(record, lang)
    if lang == "en":
        return f"vehicle {veh} license plate {plate} {err}".strip()
    return f"xe {veh} biển số {plate} {err}".strip()


def change_wav_speed(wav_bytes: bytes, speed: float) -> bytes:
    """
    Thay đổi tốc độ đọc của dữ liệu WAV bằng Overlap-Add (Hanning window)
    nhằm giữ nguyên cao độ giọng nói khi tăng tốc độ (VD: 1.25x, 1.5x, 1.75x).
    """
    speed = float(max(0.75, min(2.25, speed)))
    if not wav_bytes or abs(speed - 1.0) < 0.03:
        return wav_bytes

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw_frames = wf.readframes(wf.getnframes())

        if sampwidth != 2 or n_channels != 1 or len(raw_frames) < 400:
            return wav_bytes

        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
        win_len = max(256, int(0.045 * framerate))
        if win_len % 2 != 0:
            win_len += 1
        if len(samples) <= win_len * 2:
            return wav_bytes

        syn_hop = win_len // 2
        ana_hop = max(1, int(round(syn_hop * speed)))
        window = np.hanning(win_len).astype(np.float32)

        num_frames = max(1, (len(samples) - win_len) // ana_hop + 1)
        out_len = (num_frames - 1) * syn_hop + win_len
        out = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)

        for idx in range(num_frames):
            in_pos = idx * ana_hop
            out_pos = idx * syn_hop
            seg = samples[in_pos : in_pos + win_len]
            if len(seg) < win_len:
                break
            out[out_pos : out_pos + win_len] += seg * window
            norm[out_pos : out_pos + win_len] += window

        valid = norm > 1e-4
        out[valid] /= norm[valid]
        out_int16 = np.clip(out, -32768.0, 32767.0).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf_out:
            wf_out.setnchannels(1)
            wf_out.setsampwidth(2)
            wf_out.setframerate(framerate)
            wf_out.writeframes(out_int16.tobytes())
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Bỏ qua time-scale WAV (%s), phát nguyên bản.", exc)
        return wav_bytes


def pcm_l16_to_wav_bytes(pcm_bytes: bytes, mime_type: str) -> bytes:
    """Đóng gói audio/L16 PCM từ Gemini TTS thành bytes WAV chuẩn để phát qua winsound."""
    if mime_type and "wav" in mime_type.lower() and pcm_bytes[:4] == b"RIFF":
        return pcm_bytes

    rate = 24000
    match = re.search(r"rate=(\d+)", mime_type or "")
    if match:
        try:
            rate = int(match.group(1))
        except ValueError:
            rate = 24000

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit signed linear PCM
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class ViolationNarrator:
    """Client gọi Google AI Studio (Gemini) tạo câu đọc lỗi & giọng đọc TTS."""

    def __init__(self, settings: Optional[AISettings] = None, timeout_sec: float = 12.0):
        self.settings = settings or load_ai_settings()
        self.timeout_sec = float(timeout_sec)
        self._text_429_until: float = 0.0
        self._tts_429_until: float = 0.0

    def _headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self.settings.google_ai_api_key.strip(),
            "Content-Type": "application/json",
        }

    def _normalize_model_path(self, model_name: str) -> str:
        clean = (model_name or "").strip()
        if not clean.startswith("models/"):
            clean = f"models/{clean}"
        return clean

    def generate_sentence(self, record: ViolationRecord) -> Tuple[str, str]:
        """
        Trả về (câu đọc theo đúng ngôn ngữ & mẫu chuẩn, nguồn 'gemini' hoặc 'offline').
        """
        lang = "en" if str(self.settings.voice_language).lower() == "en" else "vi"
        fallback = format_violation_offline(record, voice_language=lang)
        if (
            not self.settings.use_ai_text_polish
            or not self.settings.google_ai_api_key.strip()
            or time.monotonic() < self._text_429_until
        ):
            return fallback, "offline"

        veh = translate_vehicle_type(record.vehicle_type, lang)
        plate = format_plate_for_reading(record.plate_number, lang)
        err = translate_violation_phrase(record, lang)

        if lang == "en":
            prompt = (
                "Write EXACTLY 1 short English sentence following the template: "
                "vehicle <type> license plate <plate> <violation>. "
                f"Data: vehicle='{veh}', plate='{plate}', "
                f"violation='{record.violation_type}', detail='{record.trigger_reason or err}'. "
                "No markdown, no quotes, must start with 'vehicle '."
            )
        else:
            prompt = (
                "Viết ĐÚNG 1 câu tiếng Việt theo mẫu: xe <loại xe> biển số <biển số> <lỗi của xe đó>. "
                f"Dữ liệu: loại xe='{veh}', biển số='{plate}', "
                f"vi phạm='{record.violation_type}', chi tiết='{record.trigger_reason or err}'. "
                "Không markdown, không dấu ngoặc kép, bắt đầu bằng 'xe '."
            )

        model_path = self._normalize_model_path(self.settings.text_model)
        url = f"{GEMINI_API_BASE}/{model_path}:generateContent"
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_sec)
            if resp.status_code != 200:
                if resp.status_code == 429:
                    self._text_429_until = time.monotonic() + 120.0
                    logger.warning("Gemini text hết hạn mức (HTTP 429), tạm chuyển chuẩn hóa cục bộ 120s.")
                else:
                    logger.warning("Gemini text HTTP %s: %s", resp.status_code, resp.text[:180])
                return fallback, "offline"

            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return fallback, "offline"
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = " ".join(str(p.get("text", "")).strip() for p in parts if p.get("text")).strip()
            text = re.sub(r"[*_`\"'\r\n]+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            lower = text.lower()
            if lang == "en":
                plate_ok = (plate == "unreadable" and "unreadable" in lower) or (plate.lower() in lower)
                if lower.startswith("vehicle ") and "license plate" in lower and plate_ok:
                    return text, "gemini"
            else:
                plate_ok = (plate == "không đọc được" and "không đọc được" in lower) or (plate.lower() in lower)
                if lower.startswith("xe ") and "biển số" in lower and plate_ok:
                    return text, "gemini"

            logger.info("Câu Gemini trả về lệch mẫu ('%s'), dùng bản chuẩn hóa offline.", text)
            return fallback, "offline"
        except Exception as exc:
            logger.warning("Lỗi gọi Gemini text (%s), fallback offline.", exc)
            return fallback, "offline"

    def synthesize_wav(self, sentence: str) -> Optional[bytes]:
        """Gọi Gemini TTS để sinh bytes WAV giọng đọc cho câu vi phạm (có cache đĩa tối đa 50 file)."""
        if (
            not self.settings.use_ai_tts_audio
            or not self.settings.google_ai_api_key.strip()
            or time.monotonic() < self._tts_429_until
        ):
            return None

        lang = "en" if str(self.settings.voice_language).lower() == "en" else "vi"
        cache_dir = Path("configs/tts_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(f"{self.settings.tts_voice}:{lang}:{sentence}".encode("utf-8")).hexdigest()[:20]
        cache_file = cache_dir / f"{cache_key}.wav"
        if cache_file.exists():
            try:
                return cache_file.read_bytes()
            except OSError:
                pass

        model_path = self._normalize_model_path(self.settings.tts_model)
        url = f"{GEMINI_API_BASE}/{model_path}:generateContent"
        if lang == "en":
            tts_prompt = f"Read this traffic alert clearly in English: {sentence}"
        else:
            tts_prompt = f"Đọc rõ ràng bằng tiếng Việt thông báo sau: {sentence}"
        payload = {
            "contents": [{"parts": [{"text": tts_prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.settings.tts_voice or "Aoede"}
                    }
                },
            },
        }
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=max(self.timeout_sec, 18.0))
            if resp.status_code != 200:
                if resp.status_code == 429:
                    self._tts_429_until = time.monotonic() + 120.0
                    logger.warning("Gemini TTS hết hạn mức (HTTP 429), chuyển sang giọng Google TTS tiếng %s.", lang.upper())
                else:
                    logger.warning("Gemini TTS HTTP %s: %s", resp.status_code, resp.text[:180])
                return None

            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = (candidates[0].get("content") or {}).get("parts") or []
            for part in parts:
                inline = part.get("inlineData") or {}
                b64 = inline.get("data")
                if b64:
                    raw = base64.b64decode(b64)
                    mime = inline.get("mimeType", "audio/L16;codec=pcm;rate=24000")
                    wav_data = pcm_l16_to_wav_bytes(raw, mime)
                    try:
                        cache_file.write_bytes(wav_data)
                        cached_files = sorted(cache_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
                        for stale in cached_files[:-50]:
                            stale.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return wav_data
            return None
        except Exception as exc:
            logger.warning("Lỗi gọi Gemini TTS (%s), fallback Google TTS.", exc)
            return None

    def synthesize_google_translate_mp3(self, sentence: str) -> Optional[bytes]:
        """
        Fallback giọng đọc chuẩn tiếng Việt (tl=vi) hoặc tiếng Anh (tl=en) qua Google Translate TTS,
        tránh tình trạng Windows không có giọng vi-VN đọc nhầm thành giọng tiếng Anh.
        """
        lang = "en" if str(self.settings.voice_language).lower() == "en" else "vi"
        cache_dir = Path("configs/tts_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(f"gtrans:{lang}:{sentence}".encode("utf-8")).hexdigest()[:20]
        cache_file = cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            try:
                return cache_file.read_bytes()
            except OSError:
                pass

        try:
            resp = requests.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "q": sentence[:200], "tl": lang, "client": "tw-ob"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8.0,
            )
            if resp.status_code == 200 and len(resp.content) > 512:
                try:
                    cache_file.write_bytes(resp.content)
                    cached_mp3 = sorted(cache_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
                    for stale in cached_mp3[:-50]:
                        stale.unlink(missing_ok=True)
                except OSError:
                    pass
                return resp.content
            logger.warning("Google Translate TTS HTTP %s", resp.status_code)
            return None
        except Exception as exc:
            logger.warning("Không gọi được Google Translate TTS (%s).", exc)
            return None

    @staticmethod
    def _play_mp3_winmm(mp3_bytes: bytes, speech_rate: float = 1.0) -> bool:
        """Phát MP3 bằng winmm.dll tích hợp sẵn của Windows, hỗ trợ tăng/giảm tốc độ đọc."""
        tmp_path = ""
        alias = f"traffic_tts_{int(time.time() * 1000) % 100000}"
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(mp3_bytes)
                tmp_path = tf.name

            winmm = ctypes.windll.winmm
            buf = ctypes.create_unicode_buffer(256)
            err = winmm.mciSendStringW(f'open "{tmp_path}" type mpegvideo alias {alias}', buf, 255, 0)
            if err != 0:
                return False

            mci_speed = max(750, min(2000, int(round(float(speech_rate) * 1000.0))))
            winmm.mciSendStringW(f"set {alias} speed {mci_speed}", buf, 255, 0)
            play_err = winmm.mciSendStringW(f"play {alias} wait", buf, 255, 0)
            winmm.mciSendStringW(f"close {alias}", None, 0, 0)
            return play_err == 0
        except Exception as exc:
            logger.warning("Lỗi phát MP3 qua winmm: %s", exc)
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def play_audio(
        wav_bytes: Optional[bytes],
        fallback_text: str,
        speech_rate: float = 1.0,
        voice_language: str = "vi",
        mp3_bytes: Optional[bytes] = None,
    ) -> str:
        """
        Thứ tự ưu tiên phát âm thanh:
          1. Gemini TTS WAV (qua winsound + tăng tốc OLA)
          2. Google TTS MP3 chuẩn tiếng Việt/Anh (qua winmm + set speed)
          3. Windows SAPI (CHỈ khi Windows có cài sẵn voice đúng ngôn ngữ VI/EN)
        """
        if wav_bytes:
            try:
                import winsound

                adjusted_wav = change_wav_speed(wav_bytes, speech_rate)
                winsound.PlaySound(adjusted_wav, winsound.SND_MEMORY)
                return "gemini_tts"
            except Exception as exc:
                logger.warning("Không phát được WAV qua winsound (%s).", exc)

        if mp3_bytes and ViolationNarrator._play_mp3_winmm(mp3_bytes, speech_rate=speech_rate):
            return f"google_tts_{voice_language}"

        # Tier 3: Windows SAPI offline — chỉ dùng nếu có giọng đúng ngôn ngữ (tránh giọng EN đọc tiếng Việt)
        lang = "en" if str(voice_language).lower() == "en" else "vi"
        target_culture = "en-US" if lang == "en" else "vi-VN"
        try:
            safe_b64 = base64.b64encode(fallback_text.encode("utf-8")).decode("ascii")
            sapi_rate = max(-4, min(8, int(round((float(speech_rate) - 1.0) * 6.0))))
            ps_cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like '{target_culture[:2]}*' }} | Select-Object -First 1; "
                "if (-not $v) { exit 2 }; "
                "$s.SelectVoice($v.VoiceInfo.Name); "
                f"$s.Rate = {sapi_rate}; "
                f"$t = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{safe_b64}')); "
                "$s.Speak($t)"
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                timeout=12,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                check=False,
            )
            if proc.returncode == 0:
                return "windows_sapi"
            logger.warning(
                "Windows chưa cài giọng SAPI '%s' và không tải được TTS trực tuyến — chỉ hiển thị cảnh báo chữ.",
                target_culture,
            )
            return "text_only"
        except Exception as exc:
            logger.warning("Fallback Windows SAPI thất bại: %s", exc)
            return "text_only"


class ViolationVoiceWorker(QThread):
    """QThread đọc lỗi vi phạm tuần tự ở nền, không làm đơ UI hay gián đoạn video."""

    narration_ready = pyqtSignal(str, str)  # (câu đọc, phương thức phát/nguồn)

    def __init__(self, settings: Optional[AISettings] = None, parent=None):
        super().__init__(parent)
        self.settings = settings or load_ai_settings()
        self.narrator = ViolationNarrator(self.settings)
        self._queue: queue.Queue[Tuple[ViolationRecord, bool]] = queue.Queue()
        self._running = True
        self._last_spoken_at: Dict[Tuple[str, int, str], float] = {}

    def update_settings(self, new_settings: AISettings) -> None:
        self.settings = new_settings
        self.narrator.settings = new_settings
        save_ai_settings(new_settings)

    def set_enabled(self, enabled: bool) -> None:
        self.settings.voice_enabled = bool(enabled)
        save_ai_settings(self.settings)

    def set_api_key(self, api_key: str) -> None:
        clean = (api_key or "").strip()
        if clean:
            self.settings.google_ai_api_key = clean
            self.narrator.settings = self.settings
            save_ai_settings(self.settings)

    def enqueue(self, record: ViolationRecord, force: bool = False) -> None:
        if not self._running:
            return
        if not force and not self.settings.voice_enabled:
            return

        now = time.time()
        dedup_key = (record.camera_id, int(record.track_id), record.violation_type)
        if not force:
            last = self._last_spoken_at.get(dedup_key, 0.0)
            if now - last < float(self.settings.cooldown_seconds):
                return
        self._last_spoken_at[dedup_key] = now

        while self._queue.qsize() >= 4:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put((record, force))

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        while self._running:
            try:
                record, force = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if not self._running:
                break
            if not force and not self.settings.voice_enabled:
                continue

            sentence, text_src = self.narrator.generate_sentence(record)
            self.narration_ready.emit(sentence, text_src)

            if self._running and (force or self.settings.voice_enabled):
                wav_bytes = self.narrator.synthesize_wav(sentence)
                mp3_bytes = None if wav_bytes else self.narrator.synthesize_google_translate_mp3(sentence)
                audio_src = self.narrator.play_audio(
                    wav_bytes,
                    sentence,
                    speech_rate=float(getattr(self.settings, "speech_rate", 1.0)),
                    voice_language=str(getattr(self.settings, "voice_language", "vi")),
                    mp3_bytes=mp3_bytes,
                )
                logger.info(
                    "Đã đọc lỗi [%s/%s | %.2fx | %s]: %s",
                    text_src,
                    audio_src,
                    float(getattr(self.settings, "speech_rate", 1.0)),
                    getattr(self.settings, "voice_language", "vi"),
                    sentence,
                )
