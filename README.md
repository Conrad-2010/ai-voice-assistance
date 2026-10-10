# Local AI Voice Assistant

A lightweight, extensible voice assistant that runs on your own hardware and controls IoT devices through natural language.

## Features

- **Local speech recognition** (Google STT by default; can be swapped for offline Sphinx)
- **Offline text-to-speech** via `pyttsx3`
- **Modular device control** — add Tasmota plugs, Home Assistant entities, or your own custom device classes
- **Optional wake word** (e.g., "computer")
- **Console mode** for testing without a microphone
- **Zero cloud dependencies** except optional speech recognition API

## Supported Commands

| Voice Command | Action |
|---------------|--------|
| "Turn on the living room light" | Turns device ON |
| "Turn off the bedroom fan" | Turns device OFF |
| "Lock the front door" | Sends ON to lock device |
| "Unlock the front door" | Sends OFF to lock device |
| "Status of the garage" | Reports device status |
| "Help" | Lists example commands |
| "Goodbye" | Exits |

## Installation

### System dependencies

**Debian / Ubuntu / Raspberry Pi OS**
```bash
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev espeak
