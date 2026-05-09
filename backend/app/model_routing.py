"""Model routing helpers for OpenRouter/Qwen-only execution."""

import os
from dataclasses import dataclass
from typing import List, Optional


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_QWEN_MODEL = "qwen/qwen3.5-flash-02-23"
BALANCED_QWEN_MODEL = "qwen/qwen3.5-plus-02-15"
REVIEW_QWEN_MODEL = "qwen/qwen3.6-plus"
FALLBACK_QWEN_MODELS = [
    BALANCED_QWEN_MODEL,
    "qwen/qwen-plus",
    PRIMARY_QWEN_MODEL,
]


@dataclass(frozen=True)
class ModelRoute:
    api_key: str
    base_url: str
    model: str
    fallback_models: List[str]


def is_qwen_model(model: str) -> bool:
    return bool(model) and model.startswith("qwen/")


def is_placeholder_secret(value: Optional[str]) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return normalized in {
        "<openrouter_api_key>",
        "your_api_key",
        "your_zep_api_key",
        "",
    }


def _csv_models(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def qwen_fallback_models() -> List[str]:
    configured = _csv_models(os.environ.get("MIROFISH_FALLBACK_MODELS", ""))
    models = configured or FALLBACK_QWEN_MODELS
    non_qwen = [model for model in models if not is_qwen_model(model)]
    if non_qwen:
        raise ValueError(
            "MIROFISH_FALLBACK_MODELS must contain only qwen/... model ids"
        )
    return models


def _require_qwen(model: str, source: str) -> str:
    if not is_qwen_model(model):
        raise ValueError(f"{source} must be a qwen/... model id, got: {model}")
    return model


def resolve_model_route(
    *,
    use_boost: bool = False,
    use_review: bool = False,
    config_model: Optional[str] = None,
) -> ModelRoute:
    """Resolve a Qwen-only OpenAI-compatible route.

    The returned API key is never logged by this module.
    """

    base_url = os.environ.get("LLM_BASE_URL") or OPENROUTER_BASE_URL
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL_NAME") or config_model or PRIMARY_QWEN_MODEL
    source = "LLM_MODEL_NAME"

    if use_boost:
        boost_api_key = os.environ.get("LLM_BOOST_API_KEY")
        if boost_api_key and not is_placeholder_secret(boost_api_key):
            api_key = boost_api_key
        base_url = os.environ.get("LLM_BOOST_BASE_URL") or base_url
        model = os.environ.get("LLM_BOOST_MODEL_NAME") or model
        source = "LLM_BOOST_MODEL_NAME"

    if use_review:
        model = os.environ.get("MIROFISH_REVIEW_MODEL") or REVIEW_QWEN_MODEL
        source = "MIROFISH_REVIEW_MODEL"

    return ModelRoute(
        api_key=api_key,
        base_url=base_url,
        model=_require_qwen(model, source),
        fallback_models=qwen_fallback_models(),
    )


def apply_openai_compatible_env(route: ModelRoute) -> None:
    """Expose route settings through env vars expected by CAMEL/OASIS."""

    if route.api_key and not is_placeholder_secret(route.api_key):
        os.environ["OPENAI_API_KEY"] = route.api_key
    if route.base_url:
        os.environ["OPENAI_API_BASE_URL"] = route.base_url
        os.environ["OPENAI_BASE_URL"] = route.base_url
