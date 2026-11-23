"""Google gateway — Gemini provider + Gemini CLI via host registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_HELP = (
    "Ducky installs and updates the Gemini CLI when this plugin is installed or "
    "updated — you should not run npm yourself. Uses this provider’s Google API "
    "key when saved, or an existing Gemini CLI login."
)


_AGY_HELP = (
    "Ducky downloads Google's Antigravity CLI for you. Log in signs in with your Google "
    "account in your own browser; chats then run on your Google AI plan, no API key."
)


def _fetch_models(api_key: str, **_kw: Any) -> Any:
    from .model_fetch import fetch_models

    return fetch_models(api_key)


def _fetch_usage(api_key: str, **kw: Any) -> Any:
    from .usage import fetch_usage

    return fetch_usage(api_key, model=str(kw.get("model") or ""))


def _skills_dir() -> str:
    return str(Path.home() / ".claude" / "skills")


def register(api) -> None:
    from . import gemini_cli_adapter as gca
    from .gemini_cli_adapter import GeminiCliAdapter
    from .gemini_provider import GeminiProvider

    from .model_fetch import clear_model_cache

    api.register_llm_provider(
        "gemini",
        factory=lambda api_key, model, **kw: GeminiProvider(api_key, model, **kw),
        fetch_models=_fetch_models,
        fetch_usage=_fetch_usage,
        test_key_model="gemini-flash-latest",  # v1beta alias; a retired model falls back (test_connection)
        tool_schema="gemini",
        clear_model_cache=clear_model_cache,
        cache_mode="implicit",
        shows_thinking_effort=True,
        is_quota_error=gca.is_limit_error,
    )
    api.register_coding_agent(
        "gemini_cli",
        factory=lambda: GeminiCliAdapter(),
        aliases=["gemini", "google_gemini"],
        skills_dir=_skills_dir,
        settings_defaults={
            "enabled": True,
            "cli_path": "",
            "default_args": "",
            "auto_approve": True,
        },
        install_help=_INSTALL_HELP,
        token_provider="gemini",
        account_env=gca.account_env,
        account_login=gca.account_login,  # older cores: a terminal tab
        account_browser_login=gca.account_browser_login,
        account_status=gca.account_status,
        account_logout=gca.account_logout,
        is_limit_error=gca.is_limit_error,
    )
    from . import agy_cli
    from . import antigravity_adapter as aga

    # Settings → LLMs → Google → Antigravity → Log in: the Google page in the person's own
    # browser and a box for its code, the way Claude Code signs in.
    api.register_coding_agent(
        "antigravity",
        factory=lambda: aga.AntigravityAdapter(),
        aliases=["agy", "antigravity_cli"],
        skills_dir=_skills_dir,
        settings_defaults={"enabled": True, "cli_path": "", "default_args": "", "auto_approve": True},
        install_help=_AGY_HELP,
        login=aga.login,
        login_submit=aga.login_submit,
        login_status=aga.login_status,
        login_cancel=aga.login_cancel,
        logout=aga.logout,
        account_env=aga.account_env,
        account_browser_login=aga.account_browser_login,
        account_status=aga.account_status,
        account_logout=aga.account_logout,
        is_limit_error=aga.is_limit_error,
    )
    agy_cli.refresh_agents()  # a plugin update or reload re-reads Antigravity's models now, not in 10 minutes
    if hasattr(api, "register_panel_rpc"):  # the code box in ui/agy_login.js
        api.register_panel_rpc("agy_login_state", lambda: agy_cli.popup_state())
        api.register_panel_rpc("agy_login_submit", lambda code="": aga.popup_submit(code))
        api.register_panel_rpc("agy_login_open", lambda: agy_cli.reopen_browser())
        api.register_panel_rpc("agy_login_cancel", lambda: agy_cli.cancel_login())
    api.register_ide_hookup("antigravity", label="Antigravity")
    try:
        from .cli_update import schedule_cli_update_on_plugin_load

        schedule_cli_update_on_plugin_load()
    except Exception:
        pass
    from . import graph_nodes

    graph_nodes.register_nodes(api)
    if hasattr(api, "tool"):  # gemini_generate_image, gemini_image_models
        graph_nodes.register_tools(api)
    # hasattr: hosts older than register_media_generator just skip it
    if hasattr(api, "register_media_generator"):
        api.register_media_generator(
            "image", graph_nodes.generate_image, {"label": "Google Gemini", "secret_key": "gemini"}
        )
    api.log("Google gateway contribution active (Providers + Gemini CLI + Antigravity + Antigravity IDE)")
