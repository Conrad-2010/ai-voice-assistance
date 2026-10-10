#!/usr/bin/env python3
"""
Local AI Voice Assistant
------------------------
An extensible, local-first voice assistant that controls IoT devices
through natural language commands. No cloud dependencies required
(except optional Google STT; offline STT is possible with Sphinx).

Usage:
    python main.py              # Voice mode
    python main.py --console    # Text mode (for testing)
"""

import argparse
import logging
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyttsx3
import requests
import speech_recognition as sr

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class AssistantConfig:
    wake_word: Optional[str] = None
    language: str = "en-US"
    voice_rate: int = 175
    energy_threshold: int = 4000
    pause_threshold: float = 0.8
    listen_timeout: float = 6.0
    phrase_time_limit: float = 8.0

# -----------------------------------------------------------------------------
# Device Abstraction Layer
# -----------------------------------------------------------------------------

class Device(ABC):
    """Base class for all controllable devices."""

    def __init__(self, name: str, aliases: Optional[List[str]] = None):
        self.name = name
        self.aliases = aliases or [name]

    @abstractmethod
    def turn_on(self) -> str:
        """Execute 'on' action. Return a human-readable result."""
        pass

    @abstractmethod
    def turn_off(self) -> str:
        """Execute 'off' action. Return a human-readable result."""
        pass

    def get_status(self) -> str:
        return f"{self.name} is connected and ready."


class TasmotaHTTPDevice(Device):
    """
    Generic HTTP/REST device (Tasmota, Shelly, generic smart plugs, etc.).
    Sends GET requests like: http://<ip>/cm?cmnd=Power%20On
    """

    def __init__(
        self,
        name: str,
        ip: str,
        on_cmd: str = "Power%20On",
        off_cmd: str = "Power%20Off",
        aliases: Optional[List[str]] = None,
    ):
        super().__init__(name, aliases)
        self.base_url = f"http://{ip}/cm"
        self.on_cmd = on_cmd
        self.off_cmd = off_cmd

    def _send(self, cmd: str) -> bool:
        try:
            resp = requests.get(f"{self.base_url}?cmnd={cmd}", timeout=5)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("HTTP device error (%s): %s", self.name, exc)
            return False

    def turn_on(self) -> str:
        return f"Turning on {self.name}." if self._send(self.on_cmd) else f"Could not reach {self.name}."

    def turn_off(self) -> str:
        return f"Turning off {self.name}." if self._send(self.off_cmd) else f"Could not reach {self.name}."


class HomeAssistantDevice(Device):
    """Controls entities via the Home Assistant REST API."""

    def __init__(
        self,
        name: str,
        entity_id: str,
        ha_url: str,
        ha_token: str,
        aliases: Optional[List[str]] = None,
    ):
        super().__init__(name, aliases)
        self.entity_id = entity_id
        self.ha_url = ha_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }

    def _call_service(self, service: str) -> bool:
        domain = self.entity_id.split(".")[0]
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        try:
            resp = requests.post(
                url,
                json={"entity_id": self.entity_id},
                headers=self.headers,
                timeout=5,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Home Assistant error (%s): %s", self.name, exc)
            return False

    def turn_on(self) -> str:
        ok = self._call_service("turn_on")
        return f"Turning on {self.name}." if ok else f"Home Assistant rejected the command for {self.name}."

    def turn_off(self) -> str:
        ok = self._call_service("turn_off")
        return f"Turning off {self.name}." if ok else f"Home Assistant rejected the command for {self.name}."


# -----------------------------------------------------------------------------
# Speech I/O
# -----------------------------------------------------------------------------

class Speaker:
    def __init__(self, rate: int = 175):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        voices = self.engine.getProperty("voices")
        if voices:
            self.engine.setProperty("voice", voices[0].id)

    def say(self, text: str):
        if not text:
            return
        logger.info("Assistant: %s", text)
        self.engine.say(text)
        self.engine.runAndWait()


class Listener:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = config.energy_threshold
        self.recognizer.pause_threshold = config.pause_threshold
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
        except OSError as exc:
            logger.error("No microphone found: %s", exc)
            raise

    def listen(self) -> Optional[str]:
        try:
            with self.microphone as source:
                logger.debug("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.config.listen_timeout,
                    phrase_time_limit=self.config.phrase_time_limit,
                )
            text = self.recognizer.recognize_google(audio, language=self.config.language)
            logger.info("Heard: %s", text)
            return text.lower()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            logger.error("STT API error: %s", exc)
            return None


# -----------------------------------------------------------------------------
# Natural Language Understanding
# -----------------------------------------------------------------------------

@dataclass
class Command:
    action: str  # on | off | status | help | exit | unknown
    device_name: Optional[str] = None
    raw_text: str = ""


class CommandProcessor:
    """Regex-based intent + entity extraction. Easy to extend."""

    def __init__(self):
        self.patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\bturn (on|off)\b(?:\s+the)?\s+(.+)", re.I), "toggle"),
            (re.compile(r"\b(switch|power)\s+(on|off)\b(?:\s+the)?\s+(.+)", re.I), "toggle"),
            (re.compile(r"\b(lock|unlock)\b(?:\s+the)?\s+(.+)", re.I), "lock"),
            (re.compile(r"\bstatus (?:of )?(?:the )?(.+)", re.I), "status"),
        ]

    def parse(self, text: str) -> Command:
        text = text.strip()

        if any(w in text for w in ["goodbye", "shut down", "exit", "quit", "stop listening"]):
            return Command(action="exit", raw_text=text)

        if text in ("help", "what can you do", "commands"):
            return Command(action="help", raw_text=text)

        for pattern, intent in self.patterns:
            match = pattern.search(text)
            if not match:
                continue

            if intent == "toggle":
                action = match.group(1).lower()
                device = match.group(2).strip()
                return Command(action=action, device_name=device, raw_text=text)

            if intent == "lock":
                action = "on" if match.group(1).lower() == "lock" else "off"
                device = match.group(2).strip()
                return Command(action=action, device_name=device, raw_text=text)

            if intent == "status":
                device = match.group(1).strip()
                return Command(action="status", device_name=device, raw_text=text)

        return Command(action="unknown", raw_text=text)


class DeviceRouter:
    def __init__(self, devices: List[Device]):
        self.devices = devices

    def find(self, phrase: str) -> Optional[Device]:
        phrase = phrase.lower()
        for dev in self.devices:
            for alias in dev.aliases:
                if alias.lower() in phrase or phrase in alias.lower():
                    return dev
        return None

    def execute(self, cmd: Command) -> str:
        if cmd.action == "help":
            return (
                "You can say things like: turn on the living room light, "
                "turn off the bedroom fan, lock the front door, or status of the garage."
            )
        if cmd.action == "unknown":
            return "I didn't quite catch that. Say 'help' for examples."
        if cmd.action == "exit":
            return "Goodbye."

        device = self.find(cmd.device_name or "")
        if not device:
            return f"I couldn't find a device matching '{cmd.device_name}'."

        if cmd.action == "on":
            return device.turn_on()
        if cmd.action == "off":
            return device.turn_off()
        if cmd.action == "status":
            return device.get_status()

        return "That action isn't supported yet."


# -----------------------------------------------------------------------------
# Assistant Orchestrator
# -----------------------------------------------------------------------------

class VoiceAssistant:
    def __init__(self, config: AssistantConfig, devices: List[Device]):
        self.config = config
        self.speaker = Speaker(rate=config.voice_rate)
        self.listener = Listener(config)
        self.processor = CommandProcessor()
        self.router = DeviceRouter(devices)
        self._running = True

    def handle_text(self, text: str) -> Tuple[str, bool]:
        """Returns (response, should_exit)."""
        if not text:
            return "", False

        if self.config.wake_word:
            if self.config.wake_word.lower() not in text:
                return "", False
            text = re.sub(re.escape(self.config.wake_word), "", text, flags=re.I).strip()

        command = self.processor.parse(text)
        response = self.router.execute(command)
        should_exit = command.action == "exit"
        return response, should_exit

    def run_voice_loop(self):
        self.speaker.say("Voice assistant online.")
        while self._running:
            try:
                text = self.listener.listen()
                if not text:
                    continue
                response, should_exit = self.handle_text(text)
                if response:
                    self.speaker.say(response)
                if should_exit:
                    self._running = False
            except KeyboardInterrupt:
                self.speaker.say("Shutting down.")
                break
            except Exception:
                logger.exception("Fatal error in voice loop")
                self.speaker.say("Something went wrong. Check the logs.")

    def run_console_loop(self):
        print("Console mode. Type 'exit' to quit.")
        while self._running:
            try:
                text = input("You: ").strip().lower()
                response, should_exit = self.handle_text(text)
                if response:
                    print(f"Assistant: {response}")
                if should_exit:
                    self._running = False
            except (EOFError, KeyboardInterrupt):
                break


# -----------------------------------------------------------------------------
# Device Registration (EDIT THIS SECTION)
# -----------------------------------------------------------------------------

def load_devices() -> List[Device]:
    """
    Register your devices here.
    Supports any number of Device subclasses.
    """
    devices: List[Device] = []

    # Example 1: Tasmota / Shelly / generic HTTP plug
    # devices.append(
    #     TasmotaHTTPDevice(
    #         name="living room light",
    #         aliases=["living room", "main light", "lamp"],
    #         ip="192.168.1.50",
    #     )
    # )

    # Example 2: Home Assistant entity (uses HA_URL and HA_TOKEN env vars)
    ha_url = os.getenv("HA_URL")
    ha_token = os.getenv("HA_TOKEN")
    if ha_url and ha_token:
        devices.append(
            HomeAssistantDevice(
                name="bedroom light",
                aliases=["bedroom", "bedroom lamp", "master light"],
                entity_id="light.bedroom",
                ha_url=ha_url,
                ha_token=ha_token,
            )
        )

    if not devices:
        logger.warning("No devices configured. Edit load_devices() in main.py.")

    return devices


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local AI Voice Assistant")
    parser.add_argument("--console", action="store_true", help="Run in text-only mode")
    parser.add_argument("--wake", type=str, default=os.getenv("WAKE_WORD"), help="Wake word")
    args = parser.parse_args()

    config = AssistantConfig(wake_word=args.wake)
    devices = load_devices()
    assistant = VoiceAssistant(config, devices)

    if args.console:
        assistant.run_console_loop()
    else:
        assistant.run_voice_loop()


if __name__ == "__main__":
    main()
