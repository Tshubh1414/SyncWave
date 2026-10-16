"""
Multi-output Windows system-audio router.
- Keeps the original compact PyQt6 UI style.
- Captures the Windows default playback device via WASAPI loopback.
- Uses a separate bounded queue and output thread for each selected output.
- Device scanning is manual after the initial startup scan; use Refresh devices.
- Default-output volume slider changes only the Windows default endpoint master volume.
- Does not edit registry, drivers, startup items, or power settings.

Install:
    python -m pip install -r requirements.txt
Run:
    python main.py
"""

import sys
import time
import threading
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio

# pycaw controls the Windows Core Audio endpoint volume; it does not select
# or change the Windows default playback device.
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except Exception:
    PYCAW_AVAILABLE = False
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QSlider, QScrollArea, QPushButton, QMessageBox, QFrame
)

# Moderate defaults: lower CPU overhead than tiny 256-frame chunks while remaining responsive.
CAPTURE_CHUNK = 960                 # ~20 ms at 48 kHz: fewer wakeups, lower CPU overhead
MAX_DELAY_MS = 500
QUEUE_LIMIT_CHUNKS = 20             # bounded memory/latency per output (~400 ms at 20 ms/chunk)
RECONNECT_SECONDS = 3.0             # less frequent retries while a device is disconnected
FORMAT = pyaudio.paInt16
DTYPE = np.int16


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_loopback(pa):
    """Find the WASAPI loopback corresponding to the current default output."""
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_index = safe_int(wasapi.get("defaultOutputDevice"), -1)
    if default_index < 0:
        raise RuntimeError("Windows does not report a default WASAPI playback device.")

    default_device = pa.get_device_info_by_index(default_index)
    target_name = str(default_device.get("name", ""))

    # Prefer an exact/closest matching loopback name.
    loopbacks = list(pa.get_loopback_device_info_generator())
    for device in loopbacks:
        name = str(device.get("name", ""))
        if target_name and (name == target_name or name.startswith(target_name) or target_name in name):
            if safe_int(device.get("maxInputChannels")) > 0:
                result = dict(device)
                result["_default_index"] = default_index
                return result

    # Fall back to the package helper, but validate its result.
    try:
        device = pa.get_default_wasapi_loopback()
        if device and safe_int(device.get("maxInputChannels")) > 0:
            result = dict(device)
            result["_default_index"] = default_index
            return result
    except Exception:
        pass

    raise RuntimeError("Could not find a WASAPI loopback for the Windows default playback device.")


def convert_channels(flat_audio, source_channels, target_channels):
    """Convert interleaved int16 samples to the target channel count."""
    source_channels = max(1, safe_int(source_channels, 2))
    target_channels = max(1, safe_int(target_channels, 2))
    usable = (flat_audio.size // source_channels) * source_channels
    if usable == 0:
        return np.zeros((0, target_channels), dtype=DTYPE)

    audio = flat_audio[:usable].reshape(-1, source_channels)
    if source_channels == target_channels:
        return audio.copy()
    if target_channels == 1:
        mono = audio.astype(np.int32).sum(axis=1, keepdims=True) // source_channels
        return np.clip(mono, -32768, 32767).astype(DTYPE)
    if source_channels == 1:
        return np.repeat(audio, target_channels, axis=1)
    if target_channels < source_channels:
        return audio[:, :target_channels].copy()
    result = np.zeros((len(audio), target_channels), dtype=DTYPE)
    result[:, :source_channels] = audio
    return result


def resample_chunk(audio, source_rate, target_rate):
    """Resample a single block using linear interpolation; bounded and dependency-light."""
    source_rate = max(1, safe_int(source_rate, 48000))
    target_rate = max(1, safe_int(target_rate, source_rate))
    if len(audio) == 0 or source_rate == target_rate:
        return audio

    # Vectorized float32 interpolation avoids repeated per-channel float64 work.
    # It is a lightweight converter intended for voice/music routing, not mastering.
    out_count = max(1, int(round(len(audio) * target_rate / source_rate)))
    positions = np.arange(out_count, dtype=np.float32) * (source_rate / target_rate)
    positions = np.minimum(positions, len(audio) - 1)
    left = positions.astype(np.int32)
    right = np.minimum(left + 1, len(audio) - 1)
    fraction = (positions - left).reshape(-1, 1)
    a = audio[left].astype(np.float32)
    b = audio[right].astype(np.float32)
    out = a + (b - a) * fraction
    return np.clip(out, -32768, 32767).astype(DTYPE)


class BoundedAudioQueue:
    """Thread-safe queue that drops oldest chunks instead of growing without limit."""
    def __init__(self, max_chunks):
        self.max_chunks = max(4, int(max_chunks))
        self.items = deque()
        self.condition = threading.Condition()

    def put(self, item):
        with self.condition:
            self.items.append(item)
            while len(self.items) > self.max_chunks:
                self.items.popleft()
            self.condition.notify()

    def get(self, timeout=0.2):
        with self.condition:
            if not self.items:
                self.condition.wait(timeout)
            return self.items.popleft() if self.items else None

    def clear(self):
        with self.condition:
            self.items.clear()

    def wake(self):
        with self.condition:
            self.condition.notify_all()


class CaptureThread(threading.Thread):
    def __init__(self, engine):
        super().__init__(name="AudioCapture", daemon=True)
        self.engine = engine
        self.stop_event = threading.Event()
        self.status = "starting"

    def stop(self):
        self.stop_event.set()

    def run(self):
        pa = self.engine.pa
        while not self.stop_event.is_set():
            stream = None
            try:
                # Do not keep a WASAPI loopback stream open while no outputs are selected.
                # This avoids needless audio capture and distribution work when the app is idle.
                if not self.engine.has_active_outputs():
                    self.status = "idle"
                    self.engine.set_capture_status("Idle — select an output to start routing")
                    self.stop_event.wait(0.75)
                    continue

                with self.engine.api_lock:
                    loopback = resolve_loopback(pa)
                    default_index = loopback["_default_index"]
                    rate = max(8000, safe_int(loopback.get("defaultSampleRate"), 48000))
                    channels = max(1, min(8, safe_int(loopback.get("maxInputChannels"), 2)))
                    stream = pa.open(
                        format=FORMAT,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=safe_int(loopback.get("index")),
                        frames_per_buffer=CAPTURE_CHUNK,
                    )

                self.status = "running"
                self.engine.set_capture_status("Capturing system audio")
                while not self.stop_event.is_set() and self.engine.has_active_outputs():
                    raw = stream.read(CAPTURE_CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(raw, dtype=DTYPE).copy()
                    self.engine.distribute(frame, rate, channels)

                    # Detect a changed Windows default playback endpoint periodically.
                    self.engine.capture_counter += 1
                    if self.engine.capture_counter % 100 == 0:
                        try:
                            with self.engine.api_lock:
                                current = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                                current_index = safe_int(current.get("defaultOutputDevice"), -1)
                            if current_index != default_index:
                                break
                        except Exception:
                            pass

            except Exception as exc:
                self.status = "error"
                self.engine.set_capture_status(f"Capture error: {exc}")
                self.engine.log(f"Capture error: {exc}")
            finally:
                if self.stop_event.is_set():
                    self.engine.set_capture_status("Stopped")
                elif self.engine.has_active_outputs():
                    self.engine.set_capture_status("Reconnecting")
                else:
                    self.status = "idle"
                    self.engine.set_capture_status("Idle — select an output to start routing")
                if stream is not None:
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass

            if not self.stop_event.is_set():
                wait_seconds = RECONNECT_SECONDS if self.engine.has_active_outputs() else 0.75
                self.stop_event.wait(wait_seconds)


class OutputWorker(threading.Thread):
    def __init__(self, engine, key, name, index):
        super().__init__(name=f"Output-{key[:28]}", daemon=True)
        self.engine = engine
        self.key = key
        self.name_label = name
        self.index = index
        self.settings_lock = threading.Lock()
        self.volume = 0.8
        self.delay_ms = 0
        self.enabled = True
        self.queue = BoundedAudioQueue(QUEUE_LIMIT_CHUNKS)
        self.stop_event = threading.Event()
        self.reconnect_event = threading.Event()
        self.status = "Waiting"
        self.stream = None
        self._last_error = None
        self._test_requested = False
        self._test_lock = threading.Lock()

    def configure(self, volume=None, delay_ms=None, enabled=None):
        with self.settings_lock:
            if volume is not None:
                self.volume = max(0.0, min(1.0, float(volume)))
            if delay_ms is not None:
                self.delay_ms = max(0, min(MAX_DELAY_MS, int(delay_ms)))
            if enabled is not None:
                self.enabled = bool(enabled)

    def push(self, frame, rate, channels):
        with self.settings_lock:
            if not self.enabled or self.stop_event.is_set():
                return
        self.queue.put((frame, rate, channels))

    def request_test(self):
        with self._test_lock:
            self._test_requested = True
        self.queue.wake()

    def request_reconnect(self):
        self.reconnect_event.set()
        self.queue.wake()

    def stop(self):
        self.stop_event.set()
        self.queue.wake()

    def _find_device(self):
        """Re-resolve by name after reconnect; only use a prior index if its name still matches."""
        pa = self.engine.pa
        with self.engine.api_lock:
            try:
                count = pa.get_device_count()
            except Exception:
                return None
            for i in range(count):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if info.get("isLoopbackDevice") or safe_int(info.get("maxOutputChannels")) < 1:
                    continue
                if str(info.get("name", "")) == self.name_label:
                    return dict(info)
            try:
                info = pa.get_device_info_by_index(self.index)
                if (not info.get("isLoopbackDevice")
                        and safe_int(info.get("maxOutputChannels")) > 0
                        and str(info.get("name", "")) == self.name_label):
                    return dict(info)
            except Exception:
                pass
        return None

    def _close_stream(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _open_stream(self):
        device = self._find_device()
        if device is None:
            raise RuntimeError("Device not found. Connect it in Windows Bluetooth settings first.")

        rate = max(8000, safe_int(device.get("defaultSampleRate"), 48000))
        max_channels = max(1, safe_int(device.get("maxOutputChannels"), 2))
        channels = 1 if max_channels == 1 else 2
        block = max(256, int(rate * 0.02))  # 20 ms block; avoids very frequent audio callbacks

        with self.engine.api_lock:
            stream = self.engine.pa.open(
                format=FORMAT,
                channels=channels,
                rate=rate,
                output=True,
                output_device_index=safe_int(device.get("index")),
                frames_per_buffer=block,
            )
        self.stream = stream
        self.index = safe_int(device.get("index"), self.index)
        self.output_rate = rate
        self.output_channels = channels
        self.status = "Connected"
        self._last_error = None
        self.queue.clear()
        self.engine.log(f"Connected: {self.name_label} ({rate} Hz, {channels} channel(s))")

    def _play_test_tone(self):
        rate = getattr(self, "output_rate", 48000)
        channels = getattr(self, "output_channels", 2)
        frames = int(rate * 0.35)
        t = np.arange(frames, dtype=np.float64) / rate
        tone = (0.12 * np.sin(2 * np.pi * 660 * t) * 32767).astype(DTYPE)
        audio = np.repeat(tone[:, None], channels, axis=1)
        with self.settings_lock:
            volume = self.volume if self.enabled else 0.0
        audio = np.clip(audio.astype(np.float32) * volume, -32768, 32767).astype(DTYPE)
        self.stream.write(np.ascontiguousarray(audio).tobytes())

    def run(self):
        delay_chunks = deque()
        delay_frames = 0
        queued_frames = 0
        current_delay = -1
        next_retry = 0.0

        while not self.stop_event.is_set():
            if self.reconnect_event.is_set():
                self.reconnect_event.clear()
                self._close_stream()
                self.queue.clear()
                delay_chunks.clear()
                queued_frames = 0
                self.status = "Reconnecting"

            if self.stream is None:
                if time.monotonic() < next_retry:
                    self.stop_event.wait(0.1)
                    continue
                try:
                    self._open_stream()
                    next_retry = 0.0
                    delay_chunks.clear()
                    queued_frames = 0
                    current_delay = -1
                except Exception as exc:
                    message = str(exc)
                    self.status = "Reconnecting"
                    if message != self._last_error:
                        self.engine.log(f"{self.name_label}: {message}")
                        self._last_error = message
                    next_retry = time.monotonic() + RECONNECT_SECONDS
                    self.stop_event.wait(0.1)
                    continue

            with self._test_lock:
                test_now = self._test_requested
                self._test_requested = False
            if test_now:
                try:
                    self._play_test_tone()
                except Exception as exc:
                    self.engine.log(f"Test tone failed for {self.name_label}: {exc}")
                    self._close_stream()
                    self.status = "Reconnecting"
                continue

            item = self.queue.get(0.2)
            if item is None:
                continue

            frame, source_rate, source_channels = item
            with self.settings_lock:
                volume = self.volume if self.enabled else 0.0
                delay_ms = self.delay_ms

            desired_delay_frames = int(self.output_rate * delay_ms / 1000.0)
            if delay_ms != current_delay:
                current_delay = delay_ms
                delay_chunks.clear()
                queued_frames = 0
                delay_frames = desired_delay_frames

            try:
                audio = convert_channels(frame, source_channels, self.output_channels)
                audio = resample_chunk(audio, source_rate, self.output_rate)
                if len(audio) == 0:
                    continue

                # Delay is implemented in whole chunks; resolution is approximately 10 ms.
                delay_chunks.append(audio)
                queued_frames += len(audio)
                if queued_frames <= delay_frames:
                    audio_out = np.zeros_like(audio)
                else:
                    audio_out = delay_chunks.popleft()
                    queued_frames -= len(audio_out)

                if volume != 1.0:
                    audio_out = np.clip(audio_out.astype(np.float32) * volume, -32768, 32767).astype(DTYPE)
                self.stream.write(np.ascontiguousarray(audio_out).tobytes())

            except Exception as exc:
                self.engine.log(f"Playback error on {self.name_label}: {exc}")
                self.status = "Reconnecting"
                self._close_stream()
                self.queue.clear()
                delay_chunks.clear()
                queued_frames = 0
                next_retry = time.monotonic() + RECONNECT_SECONDS

        self._close_stream()
        self.status = "Stopped"


class DefaultOutputVolume:
    """Read/change the master volume of the current Windows default output endpoint."""
    def __init__(self):
        self._lock = threading.RLock()

    def _endpoint(self):
        if not PYCAW_AVAILABLE:
            raise RuntimeError(
                "Default-output volume control unavailable. "
                "Install pycaw and comtypes with: pip install pycaw comtypes"
            )

        device = AudioUtilities.GetSpeakers()

        # Recent pycaw versions return an AudioDevice wrapper exposing
        # EndpointVolume. Older versions may expose Activate directly.
        endpoint = getattr(device, "EndpointVolume", None)
        if endpoint is not None:
            # In current pycaw this property already returns the
            # IAudioEndpointVolume interface.
            if hasattr(endpoint, "GetMasterVolumeLevelScalar"):
                return endpoint
            try:
                return cast(endpoint, POINTER(IAudioEndpointVolume))
            except Exception:
                pass

        activate_owner = device if hasattr(device, "Activate") else getattr(device, "_dev", None)
        if activate_owner is not None and hasattr(activate_owner, "Activate"):
            interface = activate_owner.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            return cast(interface, POINTER(IAudioEndpointVolume))

        raise RuntimeError(
            "Unsupported pycaw AudioDevice API. "
            "Try updating it with: python -m pip install --upgrade pycaw comtypes"
        )

    def get(self):
        with self._lock:
            endpoint = self._endpoint()
            return max(0, min(100, round(endpoint.GetMasterVolumeLevelScalar() * 100)))

    def set(self, percent):
        percent = max(0, min(100, int(percent)))
        with self._lock:
            endpoint = self._endpoint()
            endpoint.SetMasterVolumeLevelScalar(percent / 100.0, None)


class AudioEngine:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.api_lock = threading.RLock()
        self.outputs = {}
        self.outputs_lock = threading.RLock()
        self.capture_thread = None
        self.capture_counter = 0
        self.capture_status = "Not started"
        self.ui_log_callback = None

    def log(self, message):
        print(f"[AudioRouter] {message}")
        if self.ui_log_callback:
            self.ui_log_callback(str(message))

    def set_capture_status(self, status):
        self.capture_status = status

    def has_active_outputs(self):
        """Return whether at least one output worker is currently enabled."""
        with self.outputs_lock:
            return bool(self.outputs)

    def start(self):
        self.capture_thread = CaptureThread(self)
        self.capture_thread.start()

    def distribute(self, frame, rate, channels):
        with self.outputs_lock:
            workers = list(self.outputs.values())
        for worker in workers:
            worker.push(frame, rate, channels)

    def list_render_devices(self, include_default=False):
        result = []
        try:
            with self.api_lock:
                wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_index = safe_int(wasapi.get("defaultOutputDevice"), -1)
                for index in range(self.pa.get_device_count()):
                    try:
                        info = self.pa.get_device_info_by_index(index)
                    except Exception:
                        continue
                    if info.get("isLoopbackDevice") or safe_int(info.get("maxOutputChannels")) < 1:
                        continue
                    if safe_int(info.get("hostApi"), -1) != safe_int(wasapi.get("index"), -2):
                        continue
                    if index == default_index and not include_default:
                        continue
                    result.append({"name": str(info["name"]), "index": index})
        except Exception as exc:
            self.log(f"Device scan failed: {exc}")
        return result

    def update(self, key, name, index, active, volume, delay_ms):
        with self.outputs_lock:
            worker = self.outputs.get(key)
            if active:
                if worker is None or not worker.is_alive():
                    worker = OutputWorker(self, key, name, index)
                    worker.configure(volume=volume, delay_ms=delay_ms, enabled=True)
                    self.outputs[key] = worker
                    worker.start()
                else:
                    worker.index = index
                    worker.configure(volume=volume, delay_ms=delay_ms, enabled=True)
            elif worker is not None:
                self.outputs.pop(key, None)
                worker.stop()

    def set_max_queue_ms(self, ms):
        # Queue is deliberately bounded; this slider controls no Windows setting.
        chunks = max(3, min(50, int(ms / 20)))
        with self.outputs_lock:
            for worker in self.outputs.values():
                worker.queue.max_chunks = chunks

    def shutdown(self):
        if self.capture_thread:
            self.capture_thread.stop()
        with self.outputs_lock:
            workers = list(self.outputs.values())
            self.outputs.clear()
        for worker in workers:
            worker.stop()

        # Wait before terminating PortAudio. A timeout prevents the UI from hanging forever.
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
        for worker in workers:
            worker.join(timeout=3.0)

        alive = bool(self.capture_thread and self.capture_thread.is_alive()) or any(w.is_alive() for w in workers)
        if alive:
            self.log("Warning: an audio worker did not stop promptly; leaving audio backend intact for safety.")
            return
        try:
            self.pa.terminate()
        except Exception:
            pass


class DeviceRow(QFrame):
    """A static, lightweight card for one playback output."""
    def __init__(self, key, name, on_change, on_test):
        super().__init__()
        self.setObjectName("deviceCard")
        self.key = key
        self.name = name
        self.index = -1
        self.volume = 0.8
        self.delay_ms = 0
        self.on_change = on_change
        self.on_test = on_test

        card = QVBoxLayout(self)
        card.setContentsMargins(14, 12, 14, 12)
        card.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.checkbox = QCheckBox(name)
        self.checkbox.setObjectName("deviceName")
        self.checkbox.setToolTip("Enable or disable audio routing to this device")
        self.checkbox.stateChanged.connect(self._emit)
        top.addWidget(self.checkbox, 1)

        self.status = QLabel("OFF")
        self.status.setObjectName("deviceStatus")
        self.status.setProperty("state", "off")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setMinimumWidth(88)
        top.addWidget(self.status)

        self.test_button = QPushButton("Test tone")
        self.test_button.setObjectName("secondaryButton")
        self.test_button.setToolTip("Play a short test tone through this output")
        self.test_button.clicked.connect(lambda: self.on_test(self))
        top.addWidget(self.test_button)
        card.addLayout(top)

        controls = QHBoxLayout()
        controls.setSpacing(12)

        self.vol_label = QLabel("VOLUME  80%")
        self.vol_label.setObjectName("controlLabel")
        self.vol_label.setMinimumWidth(100)
        controls.addWidget(self.vol_label)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setObjectName("deviceSlider")
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setToolTip("Volume for this routed copy of the audio")
        self.vol_slider.valueChanged.connect(self._vol)
        controls.addWidget(self.vol_slider, 2)

        self.delay_label = QLabel("DELAY  0 ms")
        self.delay_label.setObjectName("controlLabel")
        self.delay_label.setMinimumWidth(105)
        controls.addWidget(self.delay_label)
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setObjectName("deviceSlider")
        self.delay_slider.setRange(0, MAX_DELAY_MS)
        self.delay_slider.setValue(0)
        self.delay_slider.setToolTip("Add delay to help synchronize devices with different Bluetooth latency")
        self.delay_slider.valueChanged.connect(self._delay)
        controls.addWidget(self.delay_slider, 2)
        card.addLayout(controls)

    def _vol(self, value):
        self.volume = value / 100.0
        self.vol_label.setText(f"VOLUME  {value}%")
        self._emit()

    def _delay(self, value):
        self.delay_ms = value
        self.delay_label.setText(f"DELAY  {value} ms")
        self._emit()

    def _emit(self, *_):
        self.on_change(self)


class MainWindow(QMainWindow):
    # Audio workers can log from background threads. A Qt signal makes UI updates thread-safe.
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Router")
        self.resize(1060, 760)
        self.setMinimumSize(880, 620)

        self.engine = AudioEngine()
        self.engine.ui_log_callback = self.log_signal.emit
        self.log_signal.connect(self.add_log)
        self.default_volume = DefaultOutputVolume()
        self.rows = {}
        self.include_default = False
        self.log_messages = []
        self._updating_default_volume = False
        self._last_capture_display = None
        self._last_active_count = None

        self._build_ui()
        self.engine.start()

        # No device polling or animations; this timer only refreshes status labels.
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.refresh_status)
        self.status_timer.start(2000)
        self.rescan()
        self.refresh_default_volume()

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1220;
                color: #e6edf7;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel { background: transparent; }
            QLabel#brandMark {
                background: #2563eb; color: white; border-radius: 10px;
                font-size: 13pt; font-weight: 800; min-width: 42px; max-width: 42px;
                min-height: 42px; max-height: 42px;
            }
            QLabel#appTitle { font-size: 22pt; font-weight: 750; color: #f8fafc; }
            QLabel#subtitle { color: #93a4bc; font-size: 9.5pt; }
            QLabel#sectionTitle { font-size: 11pt; font-weight: 700; color: #f1f5f9; }
            QLabel#muted { color: #91a1b8; font-size: 9pt; }
            QLabel#panelLabel { color: #a9bad0; font-size: 8.5pt; font-weight: 700; }
            QFrame#panel, QFrame#deviceCard, QFrame#capturePanel, QFrame#logPanel {
                background: #111c2e;
                border: 1px solid #26364d;
                border-radius: 13px;
            }
            QFrame#deviceCard { background: #121f33; }
            QFrame#deviceCard:hover { border: 1px solid #38537a; }
            QLabel#captureStatus {
                background: #172f52; color: #93c5fd; border-radius: 8px;
                padding: 7px 11px; font-weight: 700;
            }
            QLabel#deviceStatus {
                background: #273449; color: #a6b4c7; border-radius: 7px;
                padding: 5px 9px; font-size: 8.5pt; font-weight: 700;
            }
            QLabel#deviceStatus[state="Connected"] { background: #123d35; color: #6ee7b7; }
            QLabel#deviceStatus[state="Reconnecting"] { background: #45341c; color: #fcd34d; }
            QLabel#deviceStatus[state="Stopped"] { background: #273449; color: #a6b4c7; }
            QLabel#deviceStatus[state="Waiting"] { background: #273449; color: #a6b4c7; }
            QCheckBox { spacing: 9px; color: #e6edf7; }
            QCheckBox#deviceName { font-size: 10.5pt; font-weight: 650; }
            QCheckBox::indicator {
                width: 17px; height: 17px; border-radius: 5px;
                background: #0b1220; border: 1px solid #52647c;
            }
            QCheckBox::indicator:checked { background: #3b82f6; border: 1px solid #60a5fa; }
            QSlider::groove:horizontal { height: 5px; background: #2b3b52; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
            QSlider::add-page:horizontal { background: #2b3b52; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #dbeafe; border: 2px solid #3b82f6;
                width: 12px; margin: -5px 0; border-radius: 7px;
            }
            QPushButton {
                background: #2563eb; color: white; border: 1px solid #3675f0;
                border-radius: 8px; padding: 8px 13px; font-weight: 650;
            }
            QPushButton:hover { background: #3478f6; }
            QPushButton:pressed { background: #1d4ed8; }
            QPushButton#secondaryButton {
                background: #1b2a40; color: #dbeafe; border: 1px solid #344963;
            }
            QPushButton#secondaryButton:hover { background: #263b56; }
            QScrollArea { border: none; background: transparent; }
            QScrollArea QWidget#scrollContents { background: transparent; }
            QScrollBar:vertical { background: #0b1220; width: 10px; margin: 2px; }
            QScrollBar::handle:vertical { background: #344760; min-height: 30px; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QLabel#emptyState { color: #91a1b8; padding: 28px; }
            QLabel#logText { color: #a8b7cb; font-family: Consolas; font-size: 9pt; }
        """)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(13)

        # Compact title area: static styling only (no animated effects or live meters).
        header = QHBoxLayout()
        header.setSpacing(13)
        brand = QLabel("AR")
        brand.setObjectName("brandMark")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(brand, 0, Qt.AlignmentFlag.AlignTop)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Audio Router")
        title.setObjectName("appTitle")
        subtitle = QLabel("Send system audio to multiple headphones and speakers.")
        subtitle.setObjectName("subtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)
        header.addStretch(1)
        outer.addLayout(header)

        self.capture_panel = QFrame()
        self.capture_panel.setObjectName("capturePanel")
        capture_layout = QHBoxLayout(self.capture_panel)
        capture_layout.setContentsMargins(16, 12, 16, 12)
        capture_layout.setSpacing(12)
        capture_mark = QLabel("●")
        capture_mark.setStyleSheet("color:#60a5fa; font-size:15pt;")
        capture_layout.addWidget(capture_mark)
        capture_text = QVBoxLayout()
        capture_text.setSpacing(2)
        capture_title = QLabel("SYSTEM AUDIO CAPTURE")
        capture_title.setObjectName("panelLabel")
        self.capture_detail_label = QLabel("Starting audio engine…")
        self.capture_detail_label.setObjectName("muted")
        capture_text.addWidget(capture_title)
        capture_text.addWidget(self.capture_detail_label)
        capture_layout.addLayout(capture_text, 1)
        self.capture_status_label = QLabel("STARTING")
        self.capture_status_label.setObjectName("captureStatus")
        self.capture_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        capture_layout.addWidget(self.capture_status_label)
        outer.addWidget(self.capture_panel)

        # Controls panel.
        controls_panel = QFrame()
        controls_panel.setObjectName("panel")
        controls = QVBoxLayout(controls_panel)
        controls.setContentsMargins(16, 12, 16, 12)
        controls.setSpacing(11)
        control_head = QHBoxLayout()
        control_title = QLabel("ROUTING CONTROLS")
        control_title.setObjectName("panelLabel")
        control_head.addWidget(control_title)
        control_head.addStretch(1)
        controls.addLayout(control_head)

        bar = QHBoxLayout()
        bar.setSpacing(12)
        self.inc_default = QCheckBox("Include default output")
        self.inc_default.setToolTip("Route a second copy to the current Windows default output")
        self.inc_default.stateChanged.connect(self._toggle_default)
        bar.addWidget(self.inc_default, 2)

        queue_title = QLabel("Queue")
        queue_title.setObjectName("muted")
        bar.addWidget(queue_title)
        self.buf_slider = QSlider(Qt.Orientation.Horizontal)
        self.buf_slider.setRange(60, 1000)
        self.buf_slider.setValue(400)
        self.buf_slider.setMaximumWidth(160)
        self.buf_slider.setToolTip("Maximum buffered audio; larger values tolerate interruptions but can add latency")
        self.buf_slider.valueChanged.connect(self._buf_changed)
        bar.addWidget(self.buf_slider, 1)
        self.buf_label = QLabel("400 ms")
        self.buf_label.setMinimumWidth(55)
        self.buf_label.setObjectName("muted")
        bar.addWidget(self.buf_label)

        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.setObjectName("secondaryButton")
        self.reconnect_button.clicked.connect(self.reconnect_all)
        bar.addWidget(self.reconnect_button)
        self.refresh_button = QPushButton("↻  Refresh devices")
        self.refresh_button.clicked.connect(self.refresh_all)
        bar.addWidget(self.refresh_button)
        controls.addLayout(bar)
        outer.addWidget(controls_panel)

        # Explicitly notes that this slider changes the Windows master volume.
        volume_panel = QFrame()
        volume_panel.setObjectName("panel")
        volume_layout = QVBoxLayout(volume_panel)
        volume_layout.setContentsMargins(16, 12, 16, 12)
        volume_layout.setSpacing(9)
        volume_head = QHBoxLayout()
        volume_titles = QVBoxLayout()
        volume_titles.setSpacing(2)
        vol_title = QLabel("Windows output volume")
        vol_title.setObjectName("sectionTitle")
        vol_hint = QLabel("Changes the master volume in Windows for the current default playback device.")
        vol_hint.setObjectName("muted")
        volume_titles.addWidget(vol_title)
        volume_titles.addWidget(vol_hint)
        volume_head.addLayout(volume_titles, 1)
        self.default_volume_label = QLabel("N/A")
        self.default_volume_label.setStyleSheet("font-size:15pt; font-weight:750; color:#bfdbfe;")
        self.default_volume_label.setMinimumWidth(55)
        self.default_volume_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        volume_head.addWidget(self.default_volume_label)
        volume_layout.addLayout(volume_head)

        volume_bar = QHBoxLayout()
        self.default_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.default_volume_slider.setRange(0, 100)
        self.default_volume_slider.setValue(50)
        self.default_volume_slider.setToolTip("Adjust Windows master volume for its current default playback device")
        self.default_volume_slider.valueChanged.connect(self.set_default_volume)
        volume_bar.addWidget(self.default_volume_slider, 1)
        self.default_mute_button = QPushButton("Mute")
        self.default_mute_button.setObjectName("secondaryButton")
        self.default_mute_button.clicked.connect(self.toggle_default_mute)
        volume_bar.addWidget(self.default_mute_button)
        volume_layout.addLayout(volume_bar)
        outer.addWidget(volume_panel)

        devices_heading = QHBoxLayout()
        devices_title = QLabel("Playback devices")
        devices_title.setObjectName("sectionTitle")
        devices_heading.addWidget(devices_title)
        devices_heading.addStretch(1)
        self.device_count_label = QLabel("SCANNING…")
        self.device_count_label.setObjectName("panelLabel")
        devices_heading.addWidget(self.device_count_label)
        self.active_count_label = QLabel("0 enabled")
        self.active_count_label.setObjectName("muted")
        devices_heading.addWidget(self.active_count_label)
        outer.addLayout(devices_heading)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("scrollContents")
        self.rows_layout = QVBoxLayout(inner)
        self.rows_layout.setContentsMargins(1, 1, 5, 1)
        self.rows_layout.setSpacing(10)
        self.rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.empty_state = QLabel("No compatible playback devices found. Connect a headphone or speaker in Windows, then click Refresh devices.")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setWordWrap(True)
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rows_layout.addWidget(self.empty_state)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        log_panel = QFrame()
        log_panel.setObjectName("logPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 9, 14, 9)
        log_layout.setSpacing(5)
        log_head = QHBoxLayout()
        log_title = QLabel("ACTIVITY")
        log_title.setObjectName("panelLabel")
        log_head.addWidget(log_title)
        log_head.addStretch(1)
        self.log_state_label = QLabel("Ready")
        self.log_state_label.setObjectName("muted")
        log_head.addWidget(self.log_state_label)
        log_layout.addLayout(log_head)
        self.log_label = QLabel("Ready. Connect playback devices and select the outputs you want to use.")
        self.log_label.setObjectName("logText")
        self.log_label.setWordWrap(True)
        self.log_label.setMaximumHeight(55)
        self.log_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        log_layout.addWidget(self.log_label)
        outer.addWidget(log_panel)

        self.setCentralWidget(root)

    def add_log(self, message):
        self.log_messages.append(f"{time.strftime('%H:%M:%S')}  {message}")
        self.log_messages = self.log_messages[-4:]
        self.log_label.setText("\n".join(self.log_messages))
        self.log_state_label.setText("Updated")

    def refresh_all(self):
        self.rescan()
        self.refresh_default_volume()
        self.add_log("Device list refreshed.")

    def refresh_default_volume(self):
        try:
            value = self.default_volume.get()
            self._updating_default_volume = True
            if self.default_volume_slider.value() != value:
                self.default_volume_slider.setValue(value)
            self.default_volume_label.setText(f"{value}%")
            self.default_mute_button.setText("Unmute" if value == 0 else "Mute")
            self.default_mute_button.setEnabled(True)
        except Exception as exc:
            self.default_volume_label.setText("N/A")
            self.default_mute_button.setEnabled(False)
            self.add_log(f"Default output volume unavailable: {exc}")
        finally:
            self._updating_default_volume = False

    def set_default_volume(self, value):
        self.default_volume_label.setText(f"{value}%")
        if self._updating_default_volume:
            return
        try:
            self.default_volume.set(value)
            self.default_mute_button.setText("Unmute" if value == 0 else "Mute")
        except Exception as exc:
            self.add_log(f"Could not set default output volume: {exc}")

    def toggle_default_mute(self):
        try:
            current = self.default_volume.get()
            if current > 0:
                self._volume_before_mute = current
                self.default_volume.set(0)
                self.default_volume_slider.setValue(0)
            else:
                restore = getattr(self, "_volume_before_mute", 50)
                self.default_volume.set(restore)
                self.default_volume_slider.setValue(restore)
            self.refresh_default_volume()
        except Exception as exc:
            self.add_log(f"Could not toggle default output mute: {exc}")

    def rescan(self):
        devices = self.engine.list_render_devices(self.include_default)
        counts = {}
        seen = set()
        for device in devices:
            name = device["name"]
            occurrence = counts.get(name, 0)
            counts[name] = occurrence + 1
            key = name if occurrence == 0 else f"{name}#{occurrence}"
            seen.add(key)
            row = self.rows.get(key)
            if row is None:
                row = DeviceRow(key, name, self.on_row_change, self.on_test)
                self.rows[key] = row
                self.rows_layout.addWidget(row)
            row.index = device["index"]

        for key in list(self.rows):
            if key not in seen:
                row = self.rows.pop(key)
                self.engine.update(key, row.name, row.index, False, row.volume, row.delay_ms)
                self.rows_layout.removeWidget(row)
                row.deleteLater()

        self.empty_state.setVisible(not bool(self.rows))
        self.device_count_label.setText(f"{len(self.rows)} DEVICE" + ("S" if len(self.rows) != 1 else ""))
        self._update_active_count()

    def _update_active_count(self):
        count = sum(1 for row in self.rows.values() if row.checkbox.isChecked())
        text = f"{count} enabled"
        if self.active_count_label.text() != text:
            self.active_count_label.setText(text)

    def on_row_change(self, row):
        self.engine.update(
            row.key, row.name, row.index,
            row.checkbox.isChecked(), row.volume, row.delay_ms
        )
        self._update_active_count()

    def on_test(self, row):
        if not row.checkbox.isChecked():
            QMessageBox.information(self, "Select output first", "Enable this device before playing its test tone.")
            return
        worker = self.engine.outputs.get(row.key)
        if worker:
            worker.request_test()

    def refresh_status(self):
        capture = self.engine.capture_status
        if capture != self._last_capture_display:
            self.capture_detail_label.setText(capture)
            text = "CAPTURING" if capture == "Capturing system audio" else capture.upper()
            self.capture_status_label.setText(text[:36])
            if "error" in capture.lower():
                self.capture_status_label.setStyleSheet("background:#4c2028; color:#fda4af; border-radius:8px; padding:7px 11px; font-weight:700;")
            elif "captur" in capture.lower():
                self.capture_status_label.setStyleSheet("background:#123d35; color:#6ee7b7; border-radius:8px; padding:7px 11px; font-weight:700;")
            else:
                self.capture_status_label.setStyleSheet("")
            self._last_capture_display = capture

        with self.engine.outputs_lock:
            outputs = dict(self.engine.outputs)
        for key, row in self.rows.items():
            worker = outputs.get(key)
            state = worker.status if worker else "Off"
            if row.status.text() != state.upper():
                row.status.setText(state.upper())
                row.status.setProperty("state", state)
                row.status.style().unpolish(row.status)
                row.status.style().polish(row.status)
        self._update_active_count()

    def reconnect_all(self):
        with self.engine.outputs_lock:
            workers = list(self.engine.outputs.values())
        for worker in workers:
            worker.request_reconnect()
        self.add_log("Reconnect requested for selected outputs.")

    def _toggle_default(self, *_):
        self.include_default = self.inc_default.isChecked()
        self.add_log("Default-output option changed. Click Refresh devices to apply it.")

    def _buf_changed(self, value):
        self.buf_label.setText(f"{value} ms")
        self.engine.set_max_queue_ms(value)

    def closeEvent(self, event):
        self.status_timer.stop()
        self.engine.shutdown()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("System Audio Router")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
