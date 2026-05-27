"""Model detection, config, LLM calls."""

import json
import os
from abc import ABC, abstractmethod

import requests

from memomind_config import load_config

DETECTION_ORDER = ["ollama", "lmstudio", "openai", "anthropic", "custom"]

PREFERRED_OLLAMA = [
    "llama3.2",
    "llama3.1",
    "llama3",
    "mistral",
    "mixtral",
    "phi3.5",
    "phi3",
    "gemma3",
    "gemma2",
    "qwen2.5",
    "deepseek-r1",
]

_provider = None


class NoModelFoundError(Exception):
    pass


class ModelProvider(ABC):
    name: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "model": getattr(self, "model", None)}


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()


class LMStudioProvider(ModelProvider):
    name = "lmstudio"

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url.rstrip("/")
        self.model = "local-model"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.3},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-latest"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()


class CustomProvider(ModelProvider):
    name = "custom"

    def __init__(self, endpoint: str, model: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model or "default"
        self.api_key = api_key

    def is_available(self) -> bool:
        try:
            r = requests.get(self.endpoint, timeout=2)
            return r.status_code < 500
        except requests.RequestException:
            return True  # endpoint may not support GET

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return data.get("response", str(data)).strip()


def ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def get_ollama_models() -> list[str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.RequestException:
        return []


def pick_best_ollama_model(models: list[str]) -> str:
    for pref in PREFERRED_OLLAMA:
        for m in models:
            if pref in m.lower():
                return m
    return models[0] if models else "llama3.2"


def lmstudio_running() -> bool:
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def detect_model() -> ModelProvider:
    config = load_config()

    if config.get("model_endpoint"):
        return CustomProvider(
            config["model_endpoint"],
            config.get("model_name"),
            config.get("api_key"),
        )

    if ollama_running():
        models = get_ollama_models()
        if models:
            return OllamaProvider(pick_best_ollama_model(models))

    if lmstudio_running():
        return LMStudioProvider()

    if os.environ.get("OPENAI_API_KEY") or config.get("api_key"):
        return OpenAIProvider(config.get("api_key"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()

    raise NoModelFoundError(
        "No model found. Install Ollama or set an API key. "
        "See ~/.memomind/config.json to configure manually."
    )


def get_provider(force_refresh: bool = False) -> ModelProvider:
    global _provider
    if _provider is None or force_refresh:
        _provider = detect_model()
    return _provider


def set_provider(provider: ModelProvider):
    global _provider
    _provider = provider


def configure_model(provider: str, model: str | None = None) -> ModelProvider:
    """
    Configure the in-process model provider explicitly.
    Does not persist to config; takes effect for this process only.
    """
    provider = (provider or "").lower()
    config = load_config()

    if provider == "ollama":
        models = get_ollama_models()
        if not models:
            raise ValueError("No Ollama models found.")
        chosen = model or pick_best_ollama_model(models)
        prov = OllamaProvider(chosen)
    elif provider == "lmstudio":
        base_url = "http://localhost:1234/v1"
        prov = LMStudioProvider(base_url=base_url)
        if model:
            prov.model = model
    elif provider == "openai":
        if not (os.environ.get("OPENAI_API_KEY") or config.get("api_key")):
            raise ValueError("OPENAI_API_KEY not set.")
        prov = OpenAIProvider(config.get("api_key"), model=model or "gpt-4o-mini")
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not set.")
        prov = AnthropicProvider(model=model or "claude-3-5-haiku-latest")
    elif provider == "custom":
        endpoint = config.get("model_endpoint")
        if not endpoint:
            raise ValueError("custom model_endpoint not configured.")
        prov = CustomProvider(endpoint, model=model or config.get("model_name"), api_key=config.get("api_key"))
    else:
        raise ValueError(f"Unknown provider '{provider}'.")

    set_provider(prov)
    return prov


def complete(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    return get_provider().complete(prompt, system, max_tokens)


def extract_json_from_response(text: str) -> dict | list | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
