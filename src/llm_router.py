"""Multi-provider LLM router for the GraphRAG pipeline.

Routes all LLM inference through a priority waterfall:
  1. NVIDIA NIM  — high-quality inference, 500 free credits, fast
  2. Groq        — very fast (~460 t/s), 14,400 free req/day for 8B models
  3. Gemini      — 1M context window, great for long-context / NER tasks

Cerebras has been REMOVED: both llama3.1-8b and llama3.3-70b return
404 "Model does not exist" on this account.

The router is exposed as two async helpers:
  - ``acompletion(messages, **kw)``   — chat-style call
  - ``agenerate(prompt, **kw)``       — legacy prompt-style wrapper

Both automatically retry / fail-over on 429 errors with zero wait time.
All model names and API keys are read from environment variables — nothing
is hard-coded here. Missing providers are silently skipped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration (loaded once at import time)
# ---------------------------------------------------------------------------

def _build_model_list() -> list[dict]:
    """Build the LiteLLM Router model list from available env-var keys.
    
    Priority order (highest first):
      1. NVIDIA NIM  — high quota, good quality
      2. Groq        — fast, generous free tier
      3. Gemini      — large context window, for NER
    """
    models: list[dict] = []

    # ── 1. NVIDIA NIM (PRIMARY) ──────────────────────────────────────────
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_key:
        # Primary LLM: meta/llama-3.1-8b-instruct — fast and capable
        models.append({
            "model_name": "router-llm",
            "litellm_params": {
                "model": "nvidia_nim/meta/llama-3.1-8b-instruct",
                "api_key": nvidia_key,
                "api_base": "https://integrate.api.nvidia.com/v1",
                "rpm": 40,
                "tpm": 100_000,
            },
        })
        # Also a 70B model for higher quality when needed
        models.append({
            "model_name": "router-llm",
            "litellm_params": {
                "model": "nvidia_nim/meta/llama-3.3-70b-instruct",
                "api_key": nvidia_key,
                "api_base": "https://integrate.api.nvidia.com/v1",
                "rpm": 20,
                "tpm": 60_000,
            },
        })
        # NER alias — NVIDIA has large context so good for long-doc NER
        models.append({
            "model_name": "router-ner",
            "litellm_params": {
                "model": "nvidia_nim/meta/llama-3.1-8b-instruct",
                "api_key": nvidia_key,
                "api_base": "https://integrate.api.nvidia.com/v1",
                "rpm": 40,
                "tpm": 100_000,
            },
        })
        logger.info("LLM Router: NVIDIA NIM (llama-3.1-8b + llama-3.3-70b) registered as PRIMARY")

    # ── 2. GROQ (FAST FALLBACK) ──────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        models.append({
            "model_name": "router-llm",
            "litellm_params": {
                "model": "groq/llama-3.1-8b-instant",
                "api_key": groq_key,
                "rpm": 30,
                "tpm": 6_000,
            },
        })
        # Groq 70B for relation extraction — better triple quality
        models.append({
            "model_name": "router-llm",
            "litellm_params": {
                "model": "groq/llama3-70b-8192",
                "api_key": groq_key,
                "rpm": 30,
                "tpm": 6_000,
            },
        })
        # Groq as NER fallback
        models.append({
            "model_name": "router-ner",
            "litellm_params": {
                "model": "groq/llama-3.1-8b-instant",
                "api_key": groq_key,
                "rpm": 30,
                "tpm": 6_000,
            },
        })
        logger.info("LLM Router: Groq (llama-3.1-8b-instant + llama3-70b) registered as fallback")

    # ── 3. GEMINI (LONG-CONTEXT SPECIALIST) ─────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        models.append({
            "model_name": "router-llm",
            "litellm_params": {
                "model": "gemini/gemini-2.0-flash",
                "api_key": gemini_key,
                "rpm": 15,
                "tpm": 1_000_000,
            },
        })
        # Gemini is primary for NER due to 1M context window
        models.append({
            "model_name": "router-ner",
            "litellm_params": {
                "model": "gemini/gemini-2.0-flash",
                "api_key": gemini_key,
                "rpm": 15,
                "tpm": 1_000_000,
            },
        })
        logger.info("LLM Router: Gemini 2.0 Flash registered (llm + ner, long-context specialist)")

    # ── 4. VISION (MULTIMODAL) ──────────────────────────────────────────
    vision_models_env = os.getenv("ROUTER_VISION_MODELS", "")
    if vision_models_env:
        for vmodel in vision_models_env.split(","):
            vmodel = vmodel.strip()
            if not vmodel: continue
            if "gemini" in vmodel and gemini_key:
                models.append({
                    "model_name": "router-vision",
                    "litellm_params": {
                        "model": vmodel,
                        "api_key": gemini_key,
                        "rpm": 15,
                        "tpm": 1_000_000,
                    },
                })
            elif "nvidia" in vmodel and nvidia_key:
                models.append({
                    "model_name": "router-vision",
                    "litellm_params": {
                        "model": vmodel,
                        "api_key": nvidia_key,
                        "api_base": "https://integrate.api.nvidia.com/v1",
                        "rpm": 20,
                        "tpm": 60_000,
                    },
                })
        logger.info("LLM Router: Vision models registered: %s", vision_models_env)

    if not models:
        logger.warning(
            "LLM Router: No cloud API keys found "
            "(NVIDIA_API_KEY / GROQ_API_KEY / GEMINI_API_KEY). "
            "The router will fail at inference time. "
            "Please set at least one key in your .env file."
        )
    return models


_MODEL_LIST = _build_model_list()

# Lazily initialised so import doesn't crash when litellm isn't installed yet.
_router = None


def _get_router():
    global _router
    if _router is None:
        try:
            from litellm import Router
            _router = Router(
                model_list=_MODEL_LIST,
                routing_strategy="least-busy",
                num_retries=3,
                retry_after=0.1,
                allowed_fails=2,
                timeout=30,
                cooldown_time=15,
            )
        except ImportError:
            raise ImportError(
                "litellm is not installed. Run: uv sync"
            )
    return _router


# ---------------------------------------------------------------------------
# Audit Trail — every LLM call gets logged to a structured audit log
# ---------------------------------------------------------------------------

_audit_log: list[dict] = []

def _record_audit(
    *, call_type: str, model: str, provider: str,
    prompt_tokens: int = 0, completion_tokens: int = 0,
    latency_ms: float = 0.0, success: bool = True, error: str = ""
) -> None:
    """Append a structured record to the in-memory audit log."""
    _audit_log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "call_type": call_type,
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 1),
        "success": success,
        "error": error,
    })


def get_audit_log() -> list[dict]:
    """Return a copy of the in-memory LLM audit log."""
    return list(_audit_log)


def clear_audit_log() -> None:
    """Clear the in-memory audit log."""
    _audit_log.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def acompletion(
    messages: list[dict],
    *,
    model: str = "router-llm",
    temperature: float = 0.0,
    response_format: dict | None = None,
    **kwargs: Any,
) -> str:
    """Send a chat-format request through the multi-provider router.

    Args:
        messages: OpenAI-style message list.
        model: Router model alias (``router-llm`` or ``router-ner``).
        temperature: Sampling temperature.
        response_format: Optional ``{"type": "json_object"}`` for JSON mode.

    Returns:
        The assistant message content as a plain string.
    """
    router = _get_router()
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, **kwargs}
    if response_format:
        kw["response_format"] = response_format
    
    t0 = time.monotonic()
    try:
        response = await router.acompletion(**kw)
        latency = (time.monotonic() - t0) * 1000
        provider = response.model.split('/')[0] if '/' in response.model else response.model
        usage = getattr(response, "usage", None)
        _record_audit(
            call_type=model,
            model=response.model,
            provider=provider,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency,
            success=True,
        )
        logger.debug("LLM Router: %s via %s (%.0fms)", model, response.model, latency)
        return response.choices[0].message.content or ""
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        _record_audit(
            call_type=model, model=model, provider="unknown",
            latency_ms=latency, success=False, error=str(exc)[:200],
        )
        raise


def completion(
    messages: list[dict],
    *,
    model: str = "router-llm",
    temperature: float = 0.0,
    response_format: dict | None = None,
    **kwargs: Any,
) -> str:
    """Synchronous version of acompletion."""
    router = _get_router()
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, **kwargs}
    if response_format:
        kw["response_format"] = response_format
    
    t0 = time.monotonic()
    try:
        response = router.completion(**kw)
        latency = (time.monotonic() - t0) * 1000
        provider = response.model.split('/')[0] if '/' in response.model else response.model
        usage = getattr(response, "usage", None)
        _record_audit(
            call_type=model,
            model=response.model,
            provider=provider,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency,
            success=True,
        )
        logger.debug("LLM Router (sync): %s via %s (%.0fms)", model, response.model, latency)
        return response.choices[0].message.content or ""
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        _record_audit(
            call_type=model, model=model, provider="unknown",
            latency_ms=latency, success=False, error=str(exc)[:200],
        )
        raise



async def agenerate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    json_mode: bool = False,
    model: str = "router-llm",
    image_b64: str | None = None,
    **kwargs: Any,
) -> str:
    """Prompt-style wrapper around :func:`acompletion`.

    Args:
        prompt: The full prompt text (becomes a ``user`` message).
        system_prompt: Optional system prompt text (becomes a ``system`` message).
        json_mode: If True, requests JSON output.
        model: Router model alias.
        image_b64: Optional base64 encoded image to append to the prompt.

    Returns:
        The raw response string from the LLM.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
        
    fmt = {"type": "json_object"} if json_mode else None
    return await acompletion(messages, model=model, response_format=fmt, **kwargs)


def generate(
    prompt: str,
    *,
    system_prompt: str | None = None,
    json_mode: bool = False,
    model: str = "router-llm",
    image_b64: str | None = None,
    **kwargs: Any,
) -> str:
    """Synchronous prompt-style wrapper around completion."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
        
    fmt = {"type": "json_object"} if json_mode else None
    return completion(messages, model=model, response_format=fmt, **kwargs)


def generate_sync(prompt: str, *, json_mode: bool = False, model: str = "router-llm") -> str:
    """Synchronous wrapper for use in non-async contexts (e.g. DomainDetector).

    Runs the event loop in the current thread if none is running, otherwise
    schedules a new coroutine on the existing loop.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                agenerate(prompt, json_mode=json_mode, model=model), loop
            )
            return future.result(timeout=120)
        else:
            return loop.run_until_complete(agenerate(prompt, json_mode=json_mode, model=model))
    except RuntimeError:
        return asyncio.run(agenerate(prompt, json_mode=json_mode, model=model))


def get_provider_health() -> dict:
    """Return health status of all registered providers."""
    router = _get_router()
    health = {}
    for deployment in router.model_list:
        model_id = deployment["litellm_params"]["model"]
        provider = model_id.split('/')[0]
        health[provider] = {
            "model": model_id,
            "rpm": deployment["litellm_params"].get("rpm", 0),
            "tpm": deployment["litellm_params"].get("tpm", 0),
        }
    return health


def loads_lenient(raw: str) -> dict | list | None:
    """Parse JSON from a string, tolerating surrounding prose."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"[\{\[].*[\}\]]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
