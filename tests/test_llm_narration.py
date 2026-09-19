import pytest

import llm_narration

VALID_CONTEXT = {
    "ticker": "NVDA",
    "verdict": "Hold",
    "direction": "long",
    "win_probability": 0.536,
    "min_confidence": 0.75,
    "edge_bps": 25.8,
    "execution_cost_bps": 5.0,
    "clears_cost": True,
    "clears_confidence": False,
    "regime": "Bull",
    "in_training_universe": True,
}


def test_narrate_returns_none_when_provider_not_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert llm_narration.narrate(VALID_CONTEXT) is None


def test_narrate_returns_none_when_provider_missing(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_narration.narrate(VALID_CONTEXT) is None


@pytest.mark.parametrize("missing_var", ["OLLAMA_API_BASE", "OLLAMA_API_KEY", "LLM_MODEL"])
def test_narrate_returns_none_when_config_incomplete(monkeypatch, missing_var):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://example.invalid:1234")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv(missing_var, raising=False)
    assert llm_narration.narrate(VALID_CONTEXT) is None


def test_narrate_returns_none_on_request_failure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:1")  # nothing listens here
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    assert llm_narration.narrate(VALID_CONTEXT, timeout=2.0) is None


def test_build_user_prompt_includes_all_fields():
    prompt = llm_narration._build_user_prompt(VALID_CONTEXT)
    assert "NVDA" in prompt
    assert "Hold" in prompt
    assert "53.6%" in prompt
    assert "75%" in prompt
