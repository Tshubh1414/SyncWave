# Low-CPU Multi-Headphone Audio Router (manual refresh)

Windows 10/11. Routes Windows system audio to selected playback devices with individual volume and delay controls. Includes a modern, static dark interface designed to avoid animations, frequent redraws, and background device scanning.

## Features
- Device list scans once at startup, then only when **Refresh devices** is clicked.
- **Windows default output volume** slider controls the master volume of the current Windows default playback endpoint.
- Mute/unmute button for that default endpoint.
- Per-device volume, delay, status, and test tone.
- Reconnect all selected outputs.
- Bounded audio queues and separate output workers.
- Lightweight PyQt6 UI with no animations or live visual meters; device scanning is manual and status labels refresh every 2 seconds.
- When no output is selected, the capture thread waits and closes its loopback stream instead of continuously capturing system audio.
- UI log messages are marshalled through Qt signals for thread-safe display.
- Does not change the Windows default playback device, registry, drivers, startup entries, or power plan.

**Important:** The default-output volume slider intentionally changes Windows' normal master volume for the currently selected default output. It does not switch which device is the default. The volume feature requires `pycaw` and `comtypes`, installed by `requirements.txt`.

## Install
Use Python 3.11 on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Connect headphones in Windows first. The program scans once when it opens. After connecting or disconnecting a device, click **Refresh devices**. The interface does not animate or continuously rescan devices; most CPU use while routing is active comes from the audio capture/playback workers rather than the UI. When no output is selected, the loopback stream is closed to reduce idle work.

## Notes
- Bluetooth latency and sync depend on Windows and the Bluetooth devices; delay compensation is approximate.
- The program was syntax-checked, but audio playback and volume changes must be tested on the target Windows laptop.
