"""Antigravity coding agent: headless ``agy -p --output-format stream-json`` on the person's
Google sign-in (their Google AI plan), with Ducky's ``uefn`` tools in agy's MCP config.

agy reads one MCP config for the whole PC (``~/.gemini/config/mcp_config.json``), so the
``uefn`` server written there carries no chat identity: each chat's ``DUCKY_*`` identity
rides the agy process environment, which the bridge it starts inherits. Two chats at
once never rewrite each other's file.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
from pathlib import Path
from typing import Any

from backend.agent.coding_agents.base import CodingAgentCapabilities, CodingAgentInfo, CodingAgentLaunchResult
from backend.agent.coding_agents.cli_shared import finalize_cli_turn, truncate_tool_result
from backend.agent.coding_agents.proc_exec import run_streaming_process
from backend.agent.coding_agents.settings_helpers import coding_agent_cfg

from . import agy_cli

AGENT_ID = "antigravity"
LABEL = "Antigravity"
AUTO_ROW = {"id": "auto", "name": "Auto (Antigravity picks)", "provider": LABEL}
NEEDS_LOGIN = "Sign in to Antigravity with Google: Settings → LLMs → Google → Antigravity → Log in."
LIMITED = ("Your Google AI plan hit its Antigravity limit (429 / quota). Wait and send again, or pick "
           "another model.")
STALE_SESSION_MARKERS = ("conversation not found", "unknown conversation", "invalid conversation",
                         "no conversation", "could not find conversation")
_LIMIT_RE = re.compile(r"quota exceeded|resource[_ ]exhausted|\b429\b|too many requests|rate limit"
                       r"|exhausted your (?:daily |weekly )?(?:quota|capacity|limit)", re.I)
ARG_PROMPT_MAX = 6000  # the rest of a message goes in a file: Windows caps a command line at ~32K
_RUN_ID_FLAG = "--ducky-run-id"


def is_limit_error(text: str) -> bool:
    return isinstance(text, str) and bool(_LIMIT_RE.search(text))


def mcp_config_file() -> Path:
    return Path.home() / ".gemini" / "config" / "mcp_config.json"


def _strip_run_id(args: list[Any]) -> list[Any]:
    out: list[Any] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == _RUN_ID_FLAG:
            skip = True
            continue
        if isinstance(a, str) and a.startswith(_RUN_ID_FLAG + "="):
            continue
        out.append(a)
    return out


def shared_server(server: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """(the block for agy's PC-wide config, the env this chat's agy process carries)."""
    env = {str(k): str(v) for k, v in (server.get("env") or {}).items()}
    proc_env = dict(env)
    block: dict[str, Any] = {k: v for k, v in server.items() if k not in ("env", "args")}
    block["args"] = _strip_run_id(list(server.get("args") or []))
    keep = {k: v for k, v in env.items() if not k.startswith("DUCKY_")}
    bridge = env.get("DUCKY_BRIDGE_ARGV")
    if bridge:
        try:
            keep["DUCKY_BRIDGE_ARGV"] = json.dumps(_strip_run_id(json.loads(bridge)))
        except (ValueError, TypeError):
            pass
    # ponytail: assumes agy hands its own env to the servers it starts (every MCP client
    # here does); identity in the file instead would race between chats.
    if keep:
        block["env"] = keep
    block["disabled"] = False
    return block, proc_env


def merge_mcp(host_config_path: str, target: Path | None = None) -> dict[str, str]:
    """Put Ducky's ``uefn`` server in agy's config (the person's other servers stay); returns
    the env this chat's agy process needs. The file is only rewritten when it changes."""
    try:
        servers = json.loads(Path(host_config_path).read_text(encoding="utf-8")).get("mcpServers") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    server = servers.get("uefn") or next(iter(servers.values()), None)
    if not isinstance(server, dict):
        return {}
    block, proc_env = shared_server(server)
    target = target or mcp_config_file()
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            current = {}
    except (OSError, ValueError):
        current = {}
    mine = current.setdefault("mcpServers", {})
    if not isinstance(mine, dict):
        current["mcpServers"] = mine = {}
    if mine.get("uefn") != block:
        mine["uefn"] = block
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".ducky.tmp")
        part.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(part, target)
    return proc_env


def prompt_dir(conv_id: str) -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "UEFN-Ducky" / "antigravity"
    folder = base / re.sub(r"[^\w.-]", "_", conv_id or "chat")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def build_prompt(folder: Path, system_prompt: str, prompt: str, *, resumed: bool) -> str:
    """The ``-p`` text. The system prompt (new conversations only) and a long message go in
    files in ``folder``, which agy is given with ``--add-dir``."""
    lines: list[str] = []
    system = (system_prompt or "").strip()
    if system and not resumed:
        path = folder / "instructions.md"
        path.write_text(system, encoding="utf-8")
        lines.append(f"First read {path} and follow it for this whole conversation.")
    if len(prompt) > ARG_PROMPT_MAX or "\x00" in prompt:
        path = folder / "message.md"
        path.write_text(prompt, encoding="utf-8")
        lines.append(f"The person's message is in {path}. Do what it asks.")
    else:
        lines.append(prompt)
    return "\n\n".join(lines)


def build_argv(*, binary: str, prompt: str, model: str, extra_args: str, auto_approve: bool,
               add_dir: str = "", session_id: str = "") -> list[str]:
    argv = [binary, "-p", prompt, "--output-format", "stream-json"]
    mid = (model or "").strip()
    if mid and mid.lower() not in ("auto", "default"):
        argv += ["--model", mid]
    if (session_id or "").strip():
        argv += ["--conversation", session_id.strip()]
    if auto_approve:
        argv.append("--dangerously-skip-permissions")
    if add_dir:
        argv += ["--add-dir", add_dir]
    if (extra_args or "").strip():
        argv += shlex.split(extra_args, posix=False)
    return argv


def _num(d: dict[str, Any], *keys: str) -> int:
    for k in keys:
        try:
            v = int(d.get(k) or 0)
        except (TypeError, ValueError):
            continue
        if v:
            return max(0, v)
    return 0


class AgyStream:
    """agy's stream-json: ``init`` → ``step_update``… → ``result`` (or ``error``).
    ponytail: step fields read loosely (``text``/``content``, ``tool``/``step_type``): the
    format isn't documented; an unknown step still shows as a tool row."""

    def __init__(self, conv_id: str, run_id: str, push: Any) -> None:
        self.conv_id, self.run_id, self.push = conv_id, run_id, push
        self.text_parts: list[str] = []
        self.blocks: list[dict[str, Any]] = []
        self.session_id = ""
        self.error_text = ""
        self.usage: dict[str, int] = {}
        self._open: dict[str, str] = {}
        self._closed: set[str] = set()

    def _emit(self, event: dict[str, Any]) -> None:
        self.push({**event, "conv_id": self.conv_id, "run_id": self.run_id})

    def _text(self, chunk: str) -> None:
        if chunk:
            self.text_parts.append(chunk)
            self._emit({"type": "assistant_delta", "text": chunk})

    def on_line(self, line: str) -> None:
        try:
            data = json.loads((line or "").strip())
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        et = str(data.get("type") or "").lower()
        if et == "init":
            self.session_id = str(data.get("conversation_id") or data.get("conversationId")
                                  or data.get("session_id") or self.session_id)
        elif et in ("message", "text", "assistant", "assistant_delta"):
            if str(data.get("role") or "assistant").lower() == "assistant":
                self._text(str(data.get("text") or data.get("content") or data.get("delta") or ""))
        elif et == "step_update":
            self._step(data)
        elif et == "error":
            self.error_text = str(data.get("message") or data.get("error") or "").strip() or self.error_text
        elif et == "result":
            self.session_id = str(data.get("conversation_id") or data.get("session_id") or self.session_id)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else data.get("stats")
            if isinstance(usage, dict):
                self.usage = {"input_tokens": _num(usage, "input_tokens", "inputTokens", "input"),
                              "output_tokens": _num(usage, "output_tokens", "outputTokens", "output"),
                              "cache_read_tokens": _num(usage, "cache_read_tokens", "cachedTokens", "cached")}
            if str(data.get("subtype") or data.get("status") or "").lower() in ("error", "failed"):
                self.error_text = str(data.get("error") or data.get("message") or self.error_text)
            resp = str(data.get("result") or data.get("response") or "").strip()
            if resp and not self.text_parts:
                self._text(resp)

    def _step(self, data: dict[str, Any]) -> None:
        step = data.get("step") if isinstance(data.get("step"), dict) else data
        sid = str(step.get("step_index", data.get("step_index", "")))
        kind = str(step.get("step_type") or step.get("type") or "step").lower()
        status = str(step.get("status") or "").lower()
        if kind in ("planner_response", "response", "message", "text"):
            if status in ("", "done", "completed", "complete"):
                self._text(str(step.get("text") or step.get("content") or ""))
            return
        name = str(step.get("tool") or step.get("tool_name") or step.get("name") or kind)
        if sid in self._closed:  # agy repeats a finished step's last update
            return
        if sid not in self._open:
            args = step.get("args") or step.get("input") or step.get("parameters") or {}
            self._open[sid] = name
            self._emit({"type": "tool_start", "tool": {"id": sid or name, "name": name,
                                                       "args": args if isinstance(args, dict) else {"raw": args}}})
        if status in ("done", "completed", "complete", "error", "failed", "cancelled"):
            name = self._open.pop(sid, name)
            self._closed.add(sid)
            ok = status not in ("error", "failed", "cancelled")
            result = truncate_tool_result(str(step.get("output") or step.get("result") or step.get("error") or ""))
            self.blocks.append({"type": "tool", "name": name, "id": sid, "result": result, "ok": ok})
            self._emit({"type": "tool_done", "success": ok, "tool": {"id": sid or name, "name": name, "result": result}})


def _cfg(settings: Any = None) -> dict[str, Any]:
    try:
        if settings is None:
            from frontend.settings import PanelSettings

            settings = PanelSettings.load()
        return coding_agent_cfg(settings, AGENT_ID)
    except Exception:
        return {}


class AntigravityAdapter:
    id = AGENT_ID
    label = LABEL
    capabilities = CodingAgentCapabilities(terminal_agent=True, chat_api=False, a2a=True, mcp_inject=True,
                                           needs_api_key=False, needs_cli=True, resume=True)

    def detect(self, settings: Any) -> CodingAgentInfo:
        cfg = _cfg(settings)
        enabled = bool(cfg.get("enabled", True))
        override = str(cfg.get("cli_path") or "")
        path = agy_cli.resolve_bin(override)
        logged_in: bool | None = None
        rows: list[dict[str, str]] = []
        if path:
            logged_in, rows = agy_cli.account_models(path)
            status = f"Found: {path} · " + {True: "logged in", False: "not logged in"}.get(
                logged_in, "sign-in state unknown")
        else:
            agy_cli.install_in_background()
            status = agy_cli.install_status() or "Antigravity not found — Ducky is downloading it from Google."
        info = CodingAgentInfo(id=self.id, label=self.label, enabled=enabled, available=bool(path) and enabled,
                               status=status, cli_path=path or override,
                               default_args=str(cfg.get("default_args") or ""), capabilities=self.capabilities,
                               models=[dict(AUTO_ROW), *[r for r in rows if r["id"] != "auto"]])
        info.logged_in = logged_in  # type: ignore[attr-defined]
        return info

    def launch(self, *, prompt: str, system_prompt: str, cwd: str, conv_id: str, model: str,
               mcp_config_path: str, extra_args: str, cli_path: str, env: dict[str, str], push: Any,
               session_id: str = "", run_id: str = "", cancel: threading.Event | None = None,
               timeout_s: float = 0.0, image_paths: list[str] | None = None, **_kw: Any) -> CodingAgentLaunchResult:
        del image_paths  # ponytail: -p takes no pictures; add when agy grows a flag for them
        binary = agy_cli.resolve_bin(cli_path)
        if not binary:
            push({"type": "status", "text": "Downloading Antigravity from Google…", "conv_id": conv_id, "run_id": run_id})
            agy_cli.install()
            binary = agy_cli.resolve_bin(cli_path)
            if not binary:
                return CodingAgentLaunchResult(ok=False, status="error",
                                               error=agy_cli.install_status() or "Antigravity could not be installed.")
        run_env = {**env, **(merge_mcp(mcp_config_path) if mcp_config_path else {})}
        folder = prompt_dir(conv_id)
        auto = bool(_cfg().get("auto_approve", True))
        tails: list[str] = []

        def run(sid: str) -> CodingAgentLaunchResult:
            argv = build_argv(binary=binary, prompt=build_prompt(folder, system_prompt, prompt, resumed=bool(sid)),
                              model=model, extra_args=extra_args, auto_approve=auto, add_dir=str(folder),
                              session_id=sid)
            state = AgyStream(conv_id, run_id, push)
            proc = run_streaming_process(argv=argv, cwd=cwd, env_extra=run_env, conv_id=conv_id,
                                         on_line=state.on_line, timeout_s=timeout_s, cancel=cancel, stdin_data="")
            reply = "".join(state.text_parts).strip()
            tails[:] = [proc.stderr_tail or "", proc.raw_tail or "", state.error_text]
            return finalize_cli_turn(proc=proc, reply=reply, streamed=bool(reply) or bool(state.blocks),
                                     blocks=state.blocks, session_id=sid, new_session=state.session_id or sid,
                                     usage=state.usage, agent_label=LABEL, timeout_s=timeout_s,
                                     error_text=state.error_text, stale_session_markers=STALE_SESSION_MARKERS)

        push({"type": "status", "text": "Starting Antigravity…", "conv_id": conv_id, "run_id": run_id})
        sid = (session_id or "").strip()
        result = run(sid)
        if result.ok:
            return result
        texts = (result.error or "", *tails)
        if any(agy_cli.signed_out(t) for t in texts):
            agy_cli.forget_models()
            result.status, result.error = "needs_login", NEEDS_LOGIN
            return result
        if any(is_limit_error(t) for t in texts):
            result.error = LIMITED
            return result
        if sid and not result.reply_text and not getattr(result, "blocks", None):
            result = run("")  # agy lost that conversation: once more as a new one, never a wedged chat
        return result


# ── Settings → LLMs → Google → Antigravity: Log in (code box), like Claude Code ──

def login(cwd: str = "", cli_path: str = "", push: Any = None, **_kw: Any) -> dict[str, Any]:
    return agy_cli.start_login(cli_path or str(_cfg().get("cli_path") or ""))


def login_submit(code: str = "", cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    out = agy_cli.submit_code(code)
    if out.get("logged_in"):
        _refresh()
    return out


def login_status(cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    return agy_cli.login_status()


def login_cancel(**_kw: Any) -> dict[str, Any]:
    return agy_cli.cancel_login()


def logout(cwd: str = "", cli_path: str = "", **_kw: Any) -> dict[str, Any]:
    out = agy_cli.logout(cli_path or str(_cfg().get("cli_path") or ""))
    _refresh()
    return out


def _refresh() -> None:
    agy_cli.refresh_agents()


# ── "Antigravity logins" table (core's account hooks) ──
# Same as Claude Code: the table's Log in runs the CLI's own sign-in in a terminal tab
# (`agy acp --login`), the row's Log in is the code box above.
# ponytail: agy keeps one Google sign-in per PC, so every row is that sign-in; per-account homes
# via `--gemini-dir` if people want two Google accounts.

def account_env(config_dir: str | None) -> dict[str, str]:
    return {}


def account_login(config_dir: str | None) -> dict[str, Any]:
    binary = agy_cli.resolve_bin(str(_cfg().get("cli_path") or ""))
    if not binary:
        agy_cli.install_in_background()
        raise ValueError(agy_cli.install_status() or "Ducky is downloading Antigravity. Try again in a minute.")
    agy_cli.forget_models()  # the table re-reads the sign-in when the tab closes
    return {"argv": [binary, "acp", "--login"], "env": {}}


def account_status(config_dir: str | None) -> dict[str, Any]:
    binary = agy_cli.resolve_bin(str(_cfg().get("cli_path") or ""))
    if not binary:
        return {"loggedIn": False, "error": agy_cli.install_status() or "Antigravity isn't installed yet."}
    state, _rows = agy_cli.account_models(binary)
    return {"loggedIn": state is True}


def account_logout(config_dir: str | None) -> None:
    logout()


