"""Optional LLM narrative generation — interprets computed outputs only."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def generate_llm_narrative(
    summary: pd.DataFrame,
    insights_md: str,
    quality_report: dict[str, Any],
) -> str | None:
    """Generate an LLM narrative that interprets the deterministic outputs.

    Returns None if the openai package is unavailable or OPENAI_API_KEY is not set.

    IMPORTANT: The LLM receives only the summary table and pre-computed insights,
    never raw time-series data.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)

    summary_csv = summary.to_csv(index=False)

    prompt = (
        "You are a weather data analyst. Below is a deterministic summary table and "
        "insights report computed from historical weather observations. Write a short "
        "(3-5 paragraph) narrative interpreting the key findings. Focus on:\n"
        "- Temperature extremes and trends\n"
        "- Hours above the reference temperature and their trend over time\n"
        "- Data quality concerns\n\n"
        "Do NOT invent data. Only reference numbers from the provided summary.\n\n"
        "## Summary Table (CSV)\n"
        f"```\n{summary_csv}\n```\n\n"
        "## Deterministic Insights\n"
        f"{insights_md}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )

    return response.choices[0].message.content
