"""Lamp AI hub registration and context injection helpers."""

from memomind_api import LAMP_APP_CONFIG, get_facts, init

__all__ = ["LAMP_APP_CONFIG", "get_facts", "init", "register_app"]


def register_app(apps_registry: dict):
    """Register MemoMind in Lamp's APPS dict."""
    apps_registry["memomind"] = LAMP_APP_CONFIG


def inject_context(system_prompt: str, user_id: str, topic: str | None = None) -> str:
    """Inject MemoMind facts into an app's system prompt."""
    facts = get_facts(user_id=user_id, topic=topic)
    if not facts:
        return system_prompt
    lines = ["User context from their memory:"]
    for fact in facts:
        lines.append(f"- {fact.get('topic', 'general')}: {fact.get('summary', '')}")
    return "\n".join(lines) + "\n\n" + system_prompt
