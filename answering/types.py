"""Answer dataclass produced by the answering layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from parser.schema import Citation


@dataclass
class Answer:
    query: str
    text: str  # the LLM's answer text
    citations_used: list[Citation]  # only citations the answer referenced
    citations_all: list[Citation]  # full list surfaced to the model
    model: str  # model string that produced the answer
    finish_reason: str  # "stop" / "length" / "error" / ...
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "text": self.text,
            "citations_used": [c.citation_id for c in self.citations_used],
            "model": self.model,
            "finish_reason": self.finish_reason,
            "stats": dict(self.stats),
        }
