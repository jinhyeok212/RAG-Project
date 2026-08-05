"""환경변수로 선택한 LLM provider를 호출하는 작은 클라이언트."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from urllib import error, request

from src.retrieval_utils import load_environment


class LLMClientError(RuntimeError):
    """사용자에게 그대로 안내할 수 있는 LLM 호출 오류입니다."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    timeout_seconds: float
    base_url: str = ""
    api_key: str = ""


def load_llm_config() -> LLMConfig:
    """비밀값은 설정 객체에만 두고 출력하거나 로그에 넣지 않습니다."""
    load_environment()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60").strip())
    except ValueError as exc:
        raise LLMClientError("LLM_TIMEOUT_SECONDS는 숫자여야 합니다.") from exc
    if timeout <= 0:
        raise LLMClientError("LLM_TIMEOUT_SECONDS는 0보다 커야 합니다.")

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
        if not model:
            raise LLMClientError("OLLAMA_MODEL을 설정해주세요.")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        return LLMConfig(provider, model, timeout, base_url=base_url.rstrip("/"))
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMClientError(
                "OPENAI_API_KEY가 없습니다. .env에 API 키를 설정해주세요."
            )
        if not model:
            raise LLMClientError("OPENAI_MODEL이 없습니다. 사용할 모델명을 설정해주세요.")
        return LLMConfig(provider, model, timeout, api_key=api_key)
    raise LLMClientError("LLM_PROVIDER는 ollama 또는 openai여야 합니다.")


class LLMClient:
    """LangChain 없이 Ollama/OpenAI의 HTTP API를 호출합니다."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()

    def generate(self, prompt: str) -> str:
        if self.config.provider == "ollama":
            url = f"{self.config.base_url}/api/generate"
            payload = {"model": self.config.model, "prompt": prompt, "stream": False}
            headers = {"Content-Type": "application/json"}
        else:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }

        req = request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, ConnectionError, socket.timeout, TimeoutError) as exc:
            if isinstance(exc, (socket.timeout, TimeoutError)) or "timed out" in str(exc).lower():
                raise LLMClientError("LLM 응답 시간이 초과되었습니다.") from exc
            if self.config.provider == "ollama":
                raise LLMClientError(
                    "Ollama 서버에 연결할 수 없습니다.\n"
                    "먼저 Ollama를 실행하고 모델을 내려받아주세요."
                ) from exc
            raise LLMClientError("OpenAI API에 연결할 수 없습니다. 네트워크를 확인해주세요.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMClientError("LLM 응답 형식을 읽을 수 없습니다.") from exc

        try:
            answer = (
                body.get("response", "")
                if self.config.provider == "ollama"
                else body["choices"][0]["message"]["content"]
            )
        except (KeyError, IndexError, TypeError) as exc:
            message = body.get("error", {}).get("message", "") if isinstance(body, dict) else ""
            raise LLMClientError(f"LLM 응답에 답변이 없습니다. {message}".strip()) from exc
        answer = str(answer).strip()
        if not answer:
            raise LLMClientError("LLM 응답이 비어 있습니다.")
        return answer
