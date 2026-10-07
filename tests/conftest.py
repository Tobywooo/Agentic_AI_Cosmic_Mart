import json
from dataclasses import replace

import pytest

from app.agent import ResolutionAgent
from app.config import PROJECT_ROOT, Settings
from app.memory import ExcelMemory
from app.store import StateStore


class ScriptedLLM:
    """Plays back canned model outputs. Items may be strings, dicts (JSON-encoded) or callables(messages)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def push(self, *items):
        self.script.extend(items)

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if callable(item):
            item = item(messages)
        return item if isinstance(item, str) else json.dumps(item)


def last_observation(messages) -> dict:
    content = messages[-1]["content"]
    return json.loads(content.split("\n", 1)[1].rsplit("\n", 1)[0])


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), memory_path=tmp_path / "memory.xlsx", state_path=tmp_path / "state.json",
                   seed_path=PROJECT_ROOT / "data" / "seed_data.json", follow_up_poll_seconds=3600)


@pytest.fixture
def store(settings):
    return StateStore(settings.state_path, settings.seed_path)


@pytest.fixture
def memory(settings):
    return ExcelMemory(settings.memory_path, settings.memory_max_rows)


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def agent(llm, store, memory, settings):
    return ResolutionAgent(llm, store, memory, settings)
