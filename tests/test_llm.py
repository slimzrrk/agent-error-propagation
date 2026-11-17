from types import SimpleNamespace

import openai
import pytest

from src.llm import LLMClient, LLMConfig, derive_seed


def make_error(cls):
    # Bypass the constructors, which need transport-level request/response objects.
    exc = cls.__new__(cls)
    Exception.__init__(exc, "simulated")
    return exc


class FakeCompletions:
    def __init__(self, failures=0, exc=None):
        self.failures = failures
        self.exc = exc or make_error(openai.APIConnectionError)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        if len(self.calls) <= self.failures:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Final answer: 4"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
        )


def make_client(completions, **cfg):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(LLMConfig(retry_backoff=0.0, **cfg), client=fake)


def test_chat_returns_text_and_usage():
    comp = FakeCompletions()
    out = make_client(comp, temperature=0.7).chat([{"role": "user", "content": "2+2"}], seed=3)
    assert out.text == "Final answer: 4"
    assert (out.prompt_tokens, out.completion_tokens) == (11, 5)
    assert out.attempts == 1
    assert comp.calls[0]["seed"] == 3
    assert comp.calls[0]["temperature"] == 0.7


def test_temperature_override_and_no_seed():
    comp = FakeCompletions()
    make_client(comp).chat([], temperature=0.0)
    assert comp.calls[0]["temperature"] == 0.0
    assert "seed" not in comp.calls[0]


def test_retries_then_succeeds():
    comp = FakeCompletions(failures=2)
    out = make_client(comp, max_retries=3).chat([])
    assert out.attempts == 3


def test_gives_up_after_max_retries():
    comp = FakeCompletions(failures=10)
    with pytest.raises(openai.APIConnectionError):
        make_client(comp, max_retries=2).chat([])
    assert len(comp.calls) == 3


def test_non_retryable_error_is_raised_immediately():
    comp = FakeCompletions(failures=1, exc=make_error(openai.BadRequestError))
    with pytest.raises(openai.BadRequestError):
        make_client(comp, max_retries=3).chat([])
    assert len(comp.calls) == 1


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed("test-1", "solver1", base=42) == derive_seed("test-1", "solver1", base=42)
    assert derive_seed("test-1", "solver1", base=42) != derive_seed("test-1", "solver2", base=42)
    assert derive_seed("test-1", "solver1", base=42) != derive_seed("test-1", "solver1", base=43)
    assert 0 <= derive_seed("x") < 2**31


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        LLMConfig.from_dict({"modle": "typo"})
    assert "api_key" not in LLMConfig().to_dict()
