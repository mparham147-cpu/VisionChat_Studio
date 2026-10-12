"""
models.py
=========

The single source of truth for which AI models the app can talk to.

Two responsibilities:

1. ``MODELS`` — a registry. Each entry tells the app the model's
   provider, whether it supports reasoning effort, and whether it can
   see images. Adding a new model is one dictionary entry.

2. ``build_chat_model()`` — turns a registry entry into a ready-to-use
   LangChain chat model. All provider-specific quirks live here.
"""

import os
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# "reasoning_efforts" is an empty list when the provider does not
# document adjustable reasoning. We never invent levels.
#
# "vision" is True when the model accepts images.

MODELS = {
    "GPT-4.1": {
        "id": "gpt-4.1",
        "provider": "openai",
        "reasoning_efforts": [],
        "vision": True,
    },
    "GPT-4o mini": {
        "id": "gpt-4o-mini",
        "provider": "openai",
        "reasoning_efforts": [],
        "vision": True,
    },
    "GPT-5.2": {
        "id": "gpt-5.2",
        "provider": "openai",
        "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"],
        "vision": True,
    },
    "o4-mini": {
        "id": "o4-mini",
        "provider": "openai",
        "reasoning_efforts": ["low", "medium", "high"],
        "vision": True,
    },
    "DeepSeek Flash": {
        "id": "deepseek-flash",
        "provider": "deepseek",
        "reasoning_efforts": ["none", "low", "high", "max"],
        "vision": False,
    },
    "Llama 3.2 3B — Local": {
        "id": "llama3.2:3b",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": False,
    },

    "DeepSeek R1 — Local": {
        "id": "deepseek-r1:latest",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": False,
    },

    "Gemma 3 4B — Local": {
        "id": "gemma3:4b",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": True,
    },

    "Phi-3 — Local": {
        "id": "phi3:latest",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": False,
    },

    "Gemma 3 270M — Local": {
        "id": "gemma3:270m",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": False,
    },
    "DeepSeek V4 Flash — Ollama Cloud": {
        "id": "deepseek-v4-flash:cloud",
        "provider": "ollama",
        "reasoning_efforts": [],
        "vision": False,
    },
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_enabled_model_names():
    """
    Return the list of model names the user has enabled in Settings.

    Falls back to every model in the registry the first time the app runs.
    """
    enabled = st.session_state.get("enabled_models")
    if enabled is None:
        return list(MODELS.keys())
    return enabled


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def model_selector(key, require_vision=False, require_structured_output=False,
                   include_all_models=False):
    """
    Draw the model picker plus (when available) the reasoning-effort picker.

    Parameters
    ----------
    key : str
        Unique session key for the selectbox.
    require_vision : bool
        Only show models that accept images.
    require_structured_output : bool
        Only show models whose provider supports structured output. In
        this project only OpenAI qualifies.
    include_all_models : bool
        Show the entire registry regardless of enabled-model settings.

    Returns
    -------
    tuple
        (model_name, effort_or_None, profile_dict)
    """
    available = []
    for name in (list(MODELS) if include_all_models else get_enabled_model_names()):
        profile = MODELS[name]
        if require_vision and not profile["vision"]:
            continue
        if require_structured_output and profile["provider"] != "openai":
            continue
        available.append(name)

    if not available:
        st.warning("Enable a compatible model in Settings → Models.")
        st.stop()

    if st.session_state.get(key) not in available:
        st.session_state.pop(key, None)

    name = st.selectbox(
        "Model", available, key=key,
        help=(
            "Cloud models require provider API access. "
            "Local Ollama models run through Ollama on this computer."
        )
    )
    profile = MODELS[name]

    effort = None
    if profile["reasoning_efforts"]:
        effort = st.selectbox(
            "Reasoning effort",
            profile["reasoning_efforts"],
            key=f"{key}_{name}_effort",
            help="More effort may take longer and cost more. It does not enable web search."
        )
    else:
        st.caption("This model has no adjustable reasoning effort.")

    return name, effort, profile


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_chat_model(model_name, effort):
    """
    Instantiate a LangChain chat model for the chosen model and effort.

    All provider-specific settings live in this function so a new
    provider is a matter of adding one branch at the top.
    """
    profile = MODELS[model_name]
    provider = profile["provider"]

    if provider == "openai":
        api_key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Add OPENAI_API_KEY in Settings or in a .env file.")
        kwargs = {"api_key": api_key, "timeout": 120, "max_retries": 1}
        if effort:
            kwargs["reasoning_effort"] = effort
        return init_chat_model(profile["id"], model_provider="openai", **kwargs)

    if provider == "deepseek":
        api_key = st.session_state.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Add DEEPSEEK_API_KEY in Settings or in a .env file.")
        kwargs = {
            "api_key": api_key,
            "timeout": 120,
            "max_retries": 1,
            "base_url": "https://api.deepseek.com",
            "extra_body": {"thinking": {"type": "disabled" if effort == "none" else "enabled"}},
        }
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
        return init_chat_model(profile["id"], model_provider="openai", **kwargs)
    if provider == "ollama":
        try:
            return ChatOllama(
                model=profile["id"],
                base_url="http://localhost:11434",
                temperature=0.2,
                num_ctx=8192,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Start Ollama and confirm "
                f"that '{profile['id']}' appears in 'ollama list'."
            ) from exc

    raise ValueError(f"Unsupported provider: {provider}")


def extract_answer_text(message, include_reasoning=False):
    """
    Return the visible text from a model response.

    When ``include_reasoning`` is True, only explicitly labeled provider summary text is shown in an expander.
    Raw thinking and reasoning text are not displayed.
    The app never invents reasoning; it only shows what the provider
    actually returned.
    """
    if isinstance(message.content, str):
        return message.content

    text_parts = []
    summaries = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text"):
            text_parts.append(block.get("text", ""))
        if block.get("type") == "reasoning":
            for summary in block.get("summary", []):
                if isinstance(summary, dict) and summary.get("type") == "summary_text":
                    summaries.append(summary.get("text", ""))
    if include_reasoning and summaries:
        with st.expander("Provider reasoning summary"):
            st.write("\n".join(summaries))
    return "\n".join(text_parts)
