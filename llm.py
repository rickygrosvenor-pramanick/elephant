import os
import re
from typing import Optional
from anthropic import Anthropic

def get_anthropic_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Optional fallbacks
        for fname in ("anthropic_key.txt", "key_anthropic.txt"):
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                    if line:
                        api_key = line
                        break
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY or put key in anthropic_key.txt")
    return api_key


def get_client() -> Anthropic:
    return Anthropic(api_key=get_anthropic_key())


def get_model_name(default: str = "claude-3-5-sonnet-latest") -> str:
    return os.getenv("ANTHROPIC_MODEL", default)


def judge_binary(client: Anthropic, prompt: str, model: Optional[str] = None,
                 system: str = "Judge the advice. Output only 0 or 1.", max_tokens: int = 2) -> str:
    """Call Anthropic and coerce the result to a single '0' or '1' if present.
    Returns the raw text if no 0/1 is found, or 'ERROR: ...' on exceptions.
    """
    try:
        model_name = model or get_model_name()
        resp = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        # Anthropic SDK returns a list of content blocks; collect text
        try:
            blocks = getattr(resp, "content", None)
            if isinstance(blocks, list) and blocks:
                parts = []
                for b in blocks:
                    t = getattr(b, "text", None)
                    if t:
                        parts.append(t)
                text = "".join(parts).strip()
        except Exception:
            pass
        if not text:
            text = str(resp).strip()
        m = re.search(r"[01]", text)
        return m.group(0) if m else text
    except Exception as e:
        return f"ERROR: {e}"


class AnthropicBinaryJudge:
    """Wrapper that fixes a default model on initialization and exposes a simple judge() call."""
    def __init__(self, model: Optional[str] = None):
        self.client = get_client()
        self.model = model or get_model_name()

    def judge(self, prompt: str, system: str = "Judge the advice. Output only 0 or 1.", max_tokens: int = 2) -> str:
        return judge_binary(self.client, prompt, model=self.model, system=system, max_tokens=max_tokens)


def get_fixed_model_judge(model: Optional[str] = None) -> AnthropicBinaryJudge:
    """Convenience factory for a fixed-model judge."""
    return AnthropicBinaryJudge(model=model)
