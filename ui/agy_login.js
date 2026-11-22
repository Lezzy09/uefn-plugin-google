/* Antigravity's code box. The "Antigravity logins" table's Log in opens Google in the person's
 * own browser (backend account_browser_login); Google then shows a code, and this box takes it.
 * It asks the backend only for a while after a button is pressed, never on a timer of its own. */
(() => {
  const PID = "google";
  try { window.__duckyPluginBootCleanups?.[PID]?.(); } catch { /* gone */ }

  const WATCH_MS = 30000;
  const host = () => window.__duckyPluginHost?.forPlugin?.(PID) || null;
  const call = async (method, params) => {
    try { return (await host()?.call(method, params || {})) || {}; } catch { return {}; }
  };

  let box = null;
  let watchUntil = 0;
  let timer = 0;

  const CSS = `
.agy-login-back{position:fixed;inset:0;z-index:2147483000;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center}
.agy-login{width:min(440px,calc(100vw - 32px));background:var(--card,#1b1b1f);color:var(--fg,#eee);border:1px solid var(--border,#333);border-radius:12px;padding:20px;box-shadow:0 20px 60px rgba(0,0,0,.5);font:inherit}
.agy-login h3{margin:0 0 8px;font-size:16px}
.agy-login p{margin:0 0 14px;color:var(--muted,#aaa);font-size:13px;line-height:1.45}
.agy-login input{box-sizing:border-box;width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--border,#333);background:var(--bg,#111);color:var(--fg,#eee);font:inherit}
.agy-login .agy-err{color:var(--red,#e5484d);font-size:12px;margin-top:8px;min-height:1em}
.agy-login .agy-row{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;align-items:center}
.agy-login .agy-link{margin-right:auto;background:none;border:0;color:var(--accent,#7c9cff);cursor:pointer;padding:0;font:inherit;font-size:12px}
.agy-login button.agy-btn{padding:7px 14px;border-radius:8px;border:1px solid var(--border,#333);background:transparent;color:var(--fg,#eee);cursor:pointer;font:inherit}
.agy-login button.agy-primary{background:var(--accent,#7c9cff);border-color:var(--accent,#7c9cff);color:#fff}
.agy-login button:disabled{opacity:.6;cursor:default}`;

  function ensureStyle() {
    let s = document.getElementById("agy-login-style");
    if (!s) {
      s = document.createElement("style");
      s.id = "agy-login-style";
      document.head.appendChild(s);
    }
    s.textContent = CSS;
  }

  function closeBox() {
    box?.remove();
    box = null;
  }

  function openBox() {
    if (box) return;
    ensureStyle();
    box = document.createElement("div");
    box.className = "agy-login-back";
    box.innerHTML = `<div class="agy-login" role="dialog" aria-modal="true" aria-label="Log in to Antigravity">
      <h3>Log in to Antigravity</h3>
      <p>Sign in with Google in the browser tab that just opened. Google then shows a code: paste it here.</p>
      <input type="text" placeholder="Paste the code from Google" autocomplete="off" spellcheck="false" />
      <div class="agy-err"></div>
      <div class="agy-row">
        <button type="button" class="agy-link" data-open>Open the Google page again</button>
        <button type="button" class="agy-btn" data-cancel>Cancel</button>
        <button type="button" class="agy-btn agy-primary" data-ok>Log in</button>
      </div></div>`;
    document.body.appendChild(box);
    const input = box.querySelector("input");
    const err = box.querySelector(".agy-err");
    const ok = box.querySelector("[data-ok]");
    const submit = async () => {
      const code = input.value.trim();
      if (!code) { err.textContent = "Paste the code first."; return; }
      ok.disabled = true;
      ok.textContent = "Checking…";
      err.textContent = "";
      const res = await call("agy_login_submit", { code });
      if (!box) return;
      if (res.logged_in) { closeBox(); return; }
      ok.disabled = false;
      ok.textContent = "Log in";
      err.textContent = res.error || "That didn't work. Try again.";
      const state = await call("agy_login_state");
      if (!state.open) err.textContent += " Press Log in in the table to start again.";
    };
    ok.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") void submit(); if (e.key === "Escape") cancel(); });
    box.querySelector("[data-open]").addEventListener("click", () => void call("agy_login_open"));
    const cancel = () => { void call("agy_login_cancel"); closeBox(); };
    box.querySelector("[data-cancel]").addEventListener("click", cancel);
    input.focus();
  }

  async function tick() {
    timer = 0;
    if (box) return; // the box is up; it handles the rest
    const state = await call("agy_login_state");
    if (state.open) { openBox(); return; }
    if (Date.now() < watchUntil) timer = window.setTimeout(tick, 1000);
  }

  // Any button press may be a Log in: look for a pending sign-in for a little while.
  const onClick = (e) => {
    if (box || !(e.target instanceof Element) || !e.target.closest("button")) return;
    watchUntil = Date.now() + WATCH_MS;
    if (!timer) timer = window.setTimeout(tick, 800);
  };
  document.addEventListener("click", onClick, true);
  void tick(); // a sign-in already waiting when this script loads

  window.__duckyPluginBootCleanups = window.__duckyPluginBootCleanups || {};
  window.__duckyPluginBootCleanups[PID] = () => {
    document.removeEventListener("click", onClick, true);
    if (timer) window.clearTimeout(timer);
    closeBox();
    document.getElementById("agy-login-style")?.remove();
  };
})();
