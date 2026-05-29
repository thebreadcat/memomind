"""Single source of truth for all personality strings."""

RETURN_MESSAGES = {
    "same_day": None,
    "days_1_6": None,
    "days_7_29": None,
    "days_30_89": None,
    "days_90_364": None,
    "days_365_plus": None,
}

CAPTURE = {
    "confirm": "Got it — want me to save this?",
    "saved": "Saved. I'll remember that.",
    "updated": "Updated. Got the new version.",
    "discarded": "No problem, I'll forget I heard it.",
    "conflict": "You mentioned this before ({date}) — want to update it?",
    "connected": "This sounds related to {thread} — adding it there.",
    "new_thread": "This feels like the start of something. Should I start a thread for it?",
}

SEARCH = {
    "from_records": "From your records, ",
    "partial": "Based on the little I know so far, ",
    "stale": "You mentioned this {time} ago — still current?",
    "contradiction": "You've said two different things about this — which is true now?",
    "not_found": "I don't have anything on that yet.",
    "thin": "I only have one entry on this — want to add more?",
    "uncertain": "I'm not completely sure, but from your records: ",
}

COLD_START = {
    "no_data": "I'm just getting to know you. What should we put down first?",
    "few_entries": "I'm still learning about you — help me get this right.",
    "prompt": "Tell me one thing about yourself and I'll remember it forever.",
}

CHEERLEADER = {
    "streak": "That's {count} times this month — best stretch you've had since {month}.",
    "milestone": "You first mentioned this {time} ago. You did it.",
    "growth": "This connects to something you said back in {month}. You've come a long way.",
    "progress": "Still working on {thread}. You've added {count} things to it.",
}

STALE = {
    "flag": "This is from {time} ago — still the case?",
    "prompt": "A few things might be worth updating.",
    "refresh": "Want to go through what's changed?",
}

ERROR = {
    "no_model": "I can't think right now — no model is connected. Check your settings.",
    "db_error": "Something's wrong with my memory. Try again in a moment.",
    "too_vague": "Can you tell me a bit more? I want to get this right.",
}

TASKS = {
    "saved": "Task saved. I'll remind you.",
    "completed": "Done — marked it off.",
    "no_due": "No due date set — I'll keep it on your list.",
    "overdue": "This was due {time} ago — still need to do it?",
    "recurring": "Got it — I'll bring this back every {frequency}.",
}

EVENTS = {
    "saved": "Event saved. Reminders are set.",
    "upcoming": "You've got {title} coming up {when}.",
    "recurring": "Repeats every {frequency} — I've got it.",
    "smart_context": "Last time: {context}",
}

REMINDERS = {
    "fired": "Reminder: {title}",
    "smart_fired": "{title} — {memory_context}",
    "add_prompt": "Want me to remind you about this?",
}

IMAGES = {
    "saved": "Got it — stored and searchable.",
    "no_vision": "Image saved. What should I know about it?",
    "extracted": "Found some text in that image — saved it.",
    "storage_warn": "Images are using {size}gb. Everything's safe — just a heads up.",
}


def return_message_for_days(days: int) -> str | None:
    if days <= 0:
        return RETURN_MESSAGES["same_day"]
    if days <= 6:
        return RETURN_MESSAGES["days_1_6"]
    if days <= 29:
        return RETURN_MESSAGES["days_7_29"].format(days=days)
    if days <= 89:
        return RETURN_MESSAGES["days_30_89"]
    if days <= 364:
        return RETURN_MESSAGES["days_90_364"]
    return RETURN_MESSAGES["days_365_plus"]
