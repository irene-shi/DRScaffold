"""Configurable chat-completions transport. No provider/model/credential defaults."""

import base64
import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from data.io import digest, write_json


def load_env(path):
    """Read KEY=value without shell execution; existing environment values win."""
    if not Path(path).is_file():
        return
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid environment assignment on line {number}")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def image_content(path):
    mime = mimetypes.guess_type(str(path))[0]
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"Use JPEG, PNG or WebP images: {path}")
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def user_message(payload, images=()):
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    content.extend(image_content(path) for path in images)
    return {"role": "user", "content": content if images else content[0]["text"]}


class Client:
    def __init__(self, config, cache_dir, roles):
        self.config = config
        self.cache = Path(cache_dir)
        self.settings = {}
        for role in roles:
            values = {}
            for name in ("BASE_URL", "API_KEY", "MODEL"):
                key = f"{role}_{name}"
                value = os.environ.get(key, "").strip()
                if not value or value.lower() == "xxx":
                    raise ValueError(f"Fill {key} in data/.env or the environment")
                values[name] = value
            if not values["BASE_URL"].startswith(("https://", "http://")):
                raise ValueError(f"Invalid {role}_BASE_URL")
            self.settings[role] = values
        if {"BLIND_A", "BLIND_B"} <= set(roles):
            a, b = self.settings["BLIND_A"], self.settings["BLIND_B"]
            if a["MODEL"] == b["MODEL"]:
                raise ValueError("BLIND_A and BLIND_B must identify two different models")

    def public_settings(self):
        # Credentials are never written to cache or run metadata.
        return {role: {"model": value["MODEL"], "base_url": value["BASE_URL"]}
                for role, value in self.settings.items()}

    def ask(self, role, prompt, payload, *, images=(), examples=(), validator=None):
        setting = self.settings[role]
        messages = [{"role": "system", "content": prompt}]
        for example in examples:
            messages.extend([user_message(example["qa"], [example["image"]]),
                             {"role": "assistant", "content": json.dumps(example["annotation"], ensure_ascii=False)}])
        messages.append(user_message(payload, images))
        body = {"model": setting["MODEL"], "messages": messages,
                "temperature": self.config["temperature"], "max_tokens": self.config["max_tokens"]}
        key = digest({"url": setting["BASE_URL"], "body": body})
        cache_path = self.cache / f"{key}.json"
        if cache_path.is_file():
            result = json.loads(cache_path.read_text(encoding="utf-8"))["response"]
            if validator:
                validator(result)
            return result
        last_error = ""
        for attempt in range(self.config["request_attempts"]):
            request = urllib.request.Request(
                setting["BASE_URL"].rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {setting['API_KEY']}", "Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                    envelope = json.load(response)
                choice = envelope["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Response truncated; increase max_tokens")
                text = choice["message"]["content"].strip()
                if text.startswith("```") and text.endswith("```"):
                    text = "\n".join(text.splitlines()[1:-1])
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object")
                if validator:
                    validator(result)
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"{role}: {last_error}; check endpoint/model/credentials") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                last_error = "Connection failure or timeout"
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                last_error = "Invalid or incomplete model response"
            else:
                # Disk failures must stop here, not trigger another paid request.
                write_json(cache_path, {"role": role, "model": setting["MODEL"], "response": result})
                return result
            if attempt + 1 < self.config["request_attempts"]:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"{role}: {last_error} after {self.config['request_attempts']} attempts; rerun with --resume")
