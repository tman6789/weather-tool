"""Optional LLM narrative for decision packets — interprets pre-computed values only.

Takes a station or compare packet dict and returns a narrative markdown string.
Returns None if openai is unavailable or OPENAI_API_KEY is not set.

Hard constraints baked into the prompt
---------------------------------------
- Do NOT compute new numbers
- Only reference metric values present in the packet
- Cite specific field values by name
"""

from __future__ import annotations

import json
import math
import os
from typing import Any


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/inf with None so json.dumps never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def generate_llm_exec_summary(packet: dict[str, Any]) -> str | None:
    """Generate an LLM narrative from a decision packet dict.

    Parameters
    ----------
    packet : station or compare packet produced by ``packet.py``.

    Returns
    -------
    Markdown string or None (if API key missing or openai not installed).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)

    packet_json = json.dumps(_sanitize(packet), indent=2, default=str)

    prompt = (
        "You are a senior mechanical engineer specializing in data-center cooling "
        "infrastructure. Below is an authoritative decision packet (JSON) computed "
        "deterministically from historical weather observations.\n\n"
        "Write a concise engineering narrative (3–5 paragraphs) interpreting the key "
        "findings for a technical audience. Structure:\n"
        "1. Design stress summary (reference tdb_p996, wb_p996, death_day candidates)\n"
        "2. Efficiency outlook (air economizer hours, WEC feasibility)\n"
        "3. Freeze / resilience risk assessment\n"
        "4. Top 1–2 prioritized recommendations\n\n"
        "HARD CONSTRAINTS — violating any of these is a critical error:\n"
        "- Do NOT compute new numbers or percentages.\n"
        "- Only reference metric values that appear verbatim in the JSON packet.\n"
        "- Cite specific field names and their values (e.g., 'wb_p996 = 80.1°F').\n"
        "- Do not add caveats about data quality unless missing_data_warning is True.\n\n"
        "## Decision Packet (JSON)\n"
        f"```json\n{packet_json}\n```\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3,
    )

    return response.choices[0].message.content
