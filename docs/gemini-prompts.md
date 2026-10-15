# Frontend Redesign Prompts (Gemini 3.8 Flash Medium)

Prompts used to redesign `frontend/index.html` into a professional, refined-dark chat UI.
Model: **Gemini 3.8 Flash Medium** (Google Antigravity, free tier).

> **Note:** these prompts are historical — the redesign is already applied. Two
> later changes are NOT in the prompts below and must survive any re-run: the
> source link label is **`📄 View Document`** (not "View source document"), and
> it must only render when `source_url` is non-null (never for the
> "I don't have information about this." refusal).

Pick ONE strategy below — do not mix them.

| Strategy | When to use |
|---|---|
| **A. Agent-mode prompt (recommended)** | Antigravity IDE — model edits the file in 6 small stages, never emits one giant response |
| **B. Single-shot prompt** | Any chat UI — outputs the whole file in one block (fits in 64k output cap, but riskier on a capped free plan) |
| **C. Three-chunk prompts** | Fallback if B truncates — one prompt per section, each ~1/3 the size |

---

## Strategy A — Agent-mode prompt (paste first)

```text
MISSION
Rewrite C:\mp\wcebot\frontend\index.html into a professional, refined-dark chat UI. The file is single/self-contained (inline <style> + inline <script>, Google Font Inter is the only external resource; no frameworks, no build steps).

EXECUTION MODE — MANDATORY (this is what guarantees completion)
- You have file tools. EDIT THE FILE IN PLACE. Do NOT print the whole file in chat. Do NOT output large code blocks unless it's a fragment you're about to write.
- Work in exactly these 6 stages, in order. After EACH stage: write the changes to disk, run `node --check` on the extracted <script> (stage 5+ only) or visually re-read the section you just wrote, confirm it's complete, then proceed.
  Stage 1: <style> — tokens, shell, header, status pill, clear button
  Stage 2: <style> — messages, bubbles, timestamps, action row, error bubble, badges, source link
  Stage 3: <style> — skeleton loader, suggestion cards, composer, char counter, toast, responsive, reduced-motion, focus rings
  Stage 4: <body> markup
  Stage 5: <script> — keep existing functions (see spec) + new helpers
  Stage 6: <script> — copy/retry/clear/counter/error-bubble + final self-check
- If any single edit would exceed ~200 lines, split it into two edits before writing.
- NEVER write "...rest omitted", "// same as before", TODOs, or partial CSS selectors. Every rule/statement you start must be finished in the same edit.
- If you are interrupted, resume at the FIRST INCOMPLETE line of the current stage. Do not restart stages 1–3 if they're already saved on disk.
- Read the current file first (Read tool) so you preserve what must not change.

MUST-KEEP BEHAVIOR (do not break these)
- const API_BASE = "http://localhost:8000"
- sendMessage(): POST fetch to `${API_BASE}/chat` with {query}; const t0 = performance.now() before fetch; const elapsedMs = performance.now() - t0 after response.json(); on !response.ok → showError with the existing 429/400/generic messages; else renderBotAnswer(data, elapsedMs); finally setLoading(false).
- renderBotAnswer(data, elapsedMs): escapeHtml(answer), \n → <br>; optional .source-link with escapeAttr(source_url); then a .badge-row containing optional <span class="cached-badge">⚡ Cached response</span> and, when elapsedMs is a number, <span class="response-time-badge">⏱ Responded in X.XXs</span> (secs<10 → 2 decimals, else 1 decimal); then appendMessage("bot", html, true).
- sendChip(el): textContent, strip leading non-word chars, set inputEl.value, sendMessage() — so suggestion cards MUST start with emoji + space.
- escapeHtml, escapeAttr, scrollToBottom, showError (4s toast), autoResize (capped 120px), handleKeydown (Enter=send, Shift+Enter=newline), appendMessage(role, html, raw), setLoading(state).
- All getElementById targets must exist; every inline onclick must have a matching global `function` declaration.

SPEC

A) CSS — REFINED DARK (keep dark palette, remove heavy glow/gamer look)
:root keeps: --bg #0d0f14, --bg-card #13161e, --bg-glass rgba(255,255,255,0.04), --border rgba(255,255,255,0.08), --accent #6c63ff, --accent-dim rgba(108,99,255,0.15), --text #e8eaf0, --text-muted #7a7f96, --user-bubble linear-gradient(135deg,#6c63ff,#9b6dff), --bot-bubble rgba(255,255,255,0.05), --bot-border rgba(108,99,255,0.25), --danger #ff5a5f, --success #4ade80, --cached #f59e0b, --radius-lg 16px, --radius-md 12px, --radius-sm 8px, --transition .2s ease.
Change --accent-glow to rgba(108,99,255,0.22). ADD --radius-xl: 20px, --focus-ring: 0 0 0 3px rgba(108,99,255,0.28). Remove the noisy multi-radial body::before (max one subtle gradient).
1. Shell: #chat-container { width: min(920px,100vw); height: 100dvh; margin: 0 auto; } — full-height app, not a small centered card.
2. #messages: flex column, gap 14px, overflow-y auto, scroll-behavior smooth, scrollbar-gutter stable, 4px thin scrollbar.
3. #chat-header: flex, glass bg, bottom border; .logo = 36px circle with inline-SVG graduation cap (no emoji); .info h1 16px/700 + p 12px muted; #status-dot restyled as a status pill (green pulsing dot + text, margin-left auto); #clear-btn = icon button, muted, hover → --danger.
4. .message: flex, max-width 88%, entrance animation (fade + 8px rise); .user align-self flex-end + row-reverse; .bot flex-start. Bot bubbles get .avatar (28px circle) beside .bubble.
5. .bubble: padding 12px 16px, radius --radius-lg with 4px tail corner on sender's bottom corner, font 14.5px/1.65, word-break break-word. .message.user .bubble = gradient, white, subtle shadow (no heavy glow). .message.bot .bubble = --bot-bubble + faint border.
6. .msg-time: 10.5px, --text-muted, margin-top 6px; right-aligned for user, left for bot.
7. .msg-actions / .action-btn: row gap 6px margin-top 8px; transparent muted 11px icon+label, hover → accent; hidden until .message:hover/:focus-within, ALWAYS visible under @media (hover: none).
8. .message.error .bubble: bg rgba(255,90,95,0.10), border rgba(255,90,95,0.35), text #ff8c8f.
9. .badge-row: flex wrap gap 6px margin-top 8px; .cached-badge (amber) and .response-time-badge (muted) as quiet 11px/6px-radius chips — keep both class names.
10. .source-link: quiet pill, hover accent.
11. #typing-indicator: hidden by default; its .bubble holds #skeleton = 3 .sk-bar shimmer bars (height 10px; widths 220/170/120px; radius 6px; 1.4s shimmer keyframes). The old .dot bouncing dots are removed.
12. #prompt-chips: grid repeat(2,1fr) gap 10px; .chip/.suggestion-card = card (glass bg, 1px --border, radius --radius-md, padding 12px 14px, text-left) + emoji label + muted .hint line; hover lifts 1px + accent border; focus ring.
13. #input-area: bordered bar; #query-input pill (radius 24px, padding 11px 16px, resize none, max-height 120px, focus = --focus-ring); #send-btn 44px CIRCLE, gradient, disabled 0.4; #send-btn.loading shows a CSS spinner ring.
14. #char-count: 11px muted, fixed min-height; .warn → #f59e0b (>450); .full → --danger (500).
15. #error-toast: quiet red tint, .show visible, sits above composer.
16. :focus-visible { box-shadow: var(--focus-ring); } on all controls; @media (prefers-reduced-motion: reduce) disable all animations/transitions.

B) MARKUP — exact ids are FIXED (JS targets them)
<div id="chat-container" role="main">
- #chat-header: .logo inline SVG cap · .info > h1 "WCE Bot" + p "Walchand College of Engineering · Official Information Assistant" · #status-dot pill (green dot + <span>Online</span>, title/aria-label "Service online") · #clear-btn <button type="button" onclick="clearChat()" aria-label="Clear conversation" title="Clear chat"> SVG trash.
- #messages (role="log" aria-label="Chat messages" aria-live="polite") containing only <div class="message bot" id="welcome-msg"><div class="bubble">👋 Hello! I'm <strong>WCE Bot</strong> — ask me about fees, syllabus, admissions or exams, all from official college documents.<br><em>What would you like to know?</em></div></div> (short, no long <br> walls).
- <div id="typing-indicator" class="message bot" role="status" aria-label="WCE Bot is typing"><div class="bubble"><div id="skeleton"><span class="sk-bar"></span><span class="sk-bar"></span><span class="sk-bar"></span></div></div></div>
- #prompt-chips (aria-label="Quick question suggestions") with 4 buttons: class="chip suggestion-card", ids chip-1..chip-4, each onclick="sendChip(this)", text starts with emoji + space then label + muted hint:
  chip-1 💰 Fee structure / "Tuition, hostel, exam fees"
  chip-2 📚 Syllabus for CS / "Semester-wise course plan"
  chip-3 📅 Admission deadlines / "Dates, eligibility, quota"
  chip-4 📄 Download prospectus / "PDF overview of the college"
- #input-area: <textarea id="query-input" placeholder="Ask about fees, syllabus, admissions…" rows="1" maxlength="500" aria-label="Type your question" onkeydown="handleKeydown(event)" oninput="autoResize(this)"></textarea> + <button id="send-btn" type="button" onclick="sendMessage()" aria-label="Send message" title="Send (Enter)"> SVG paper-plane </button> + <div id="char-count">0/500</div>
- <div id="error-toast" role="alert" aria-live="assertive"></div>
</div><!-- /chat-container -->

C) JAVASCRIPT — all functions declared with `function` (global, so inline handlers resolve)
State: const API_BASE = "http://localhost:8000"; let isLoading = false; let lastQuery = ""; DOM refs via getElementById: messagesEl(#messages), inputEl(#query-input), sendBtn(#send-btn), typingEl(#typing-indicator), chipsEl(#prompt-chips), errorToastEl(#error-toast), charCountEl(#char-count); const welcomeTpl = the #welcome-msg cloneNode(true) captured at load.
Keep verbatim in behavior: autoResize (also calls updateCount), handleKeydown, sendChip, escapeHtml, escapeAttr, scrollToBottom, showError (4s), setLoading (adds/removes "loading" class on sendBtn + aria-busy on messagesEl + toggles typingEl display flex/none), sendMessage (full logic in MUST-KEEP above; also stores lastQuery, hides chips, clears input + resets counter), renderBotAnswer (full badge logic above, PLUS appends <div class="msg-actions"><button type="button" class="action-btn" onclick="copyBubble(this)">⧉ Copy</button></div>), appendMessage (now also appends <span class="msg-time">now()</span> to every bubble).
New: now() → HH:MM 24h zero-padded; updateCount() → n+"/500" + warn/full classes; clearChat() → if isLoading return; wipe messagesEl, re-append welcomeTpl clone, show chipsEl again, hide toast, updateCount, scrollToBottom; appendErrorBubble(query) → .message.error.bot bubble with escaped "⚠ Couldn't get an answer. Check that the backend is running, then try again.", .msg-time, and .msg-actions with [Retry onclick="retrySend()"] + [Copy onclick="copyBubble(this)"]; retrySend() → if isLoading or !lastQuery return; inputEl.value = lastQuery; sendMessage(); copyBubble(btn) → clone bubble, strip .msg-actions, take innerText, navigator.clipboard.writeText(...).then(flash).catch(() => showError("Copy failed")), fallback showError("Copy failed") if no clipboard; flash(btn) → "✓ Copied" then restore after 1500ms.
Network/HTTP failure paths must call showError(...) AND appendErrorBubble(query).

DONE CRITERIA (verify before reporting back)
1. No "...", no TODOs, no missing functions anywhere in the file.
2. `node --check` passes on the extracted <script> (or at minimum: balanced braces/tags verified by re-reading the file).
3. Every id in the list above exists exactly once: chat-container, chat-header, status-dot, clear-btn, messages, welcome-msg, typing-indicator, skeleton, prompt-chips, chip-1..chip-4, input-area, query-input, send-btn, char-count, error-toast.
4. Every class used in markup has a CSS rule (esp. badge-row, cached-badge, response-time-badge, msg-time, msg-actions, action-btn, sk-bar, suggestion-card, hint, char-count warn/full, send-btn loading, error-toast show).
5. Response-time + cached badges, CORS behavior, and XSS escaping unchanged.
6. Do NOT touch any backend file (app/, tests/) — this is frontend-only.

Report back: stages completed, verification results, and any deviation from this spec.
```

**Usage:** paste as your first message. If it stops, reply `resume the current stage from the first incomplete line` — completed stages on disk are not redone.

---

## Strategy B — Single-shot prompt (one response, full file)

````text
You are rewriting the frontend of "WCE Bot", a dark-theme college information chatbot. Output the COMPLETE file `frontend/index.html` in ONE ```html code block.

## OUTPUT RULES (critical — violating these fails the task)
- The file is single/self-contained: <!DOCTYPE html> ... </html>, with inline <style> and inline <script>. Google Fonts "Inter" is the only allowed external resource. No frameworks, no build steps, no CDN JS/CSS.
- Output EVERY section in full: every CSS rule, every HTML element, every JS function. NEVER write "...rest of file", "// same as before", "/* omitted */", or any placeholder.
- No commentary outside the code block.
- All UI icons = inline SVG. Emoji allowed ONLY inside suggestion-card labels and the greeting.
- Every getElementById target must exist; every onclick handler must have a matching global `function`.

## PART A — CSS (<style>)
Design direction: REFINED DARK. Keep the existing dark palette, remove the heavy "glow/gamer" look.

Tokens in :root (keep these, add the marked ones):
--bg #0d0f14, --bg-card #13161e, --bg-glass rgba(255,255,255,0.04), --border rgba(255,255,255,0.08), --accent #6c63ff, --accent-dim rgba(108,99,255,0.15), --text #e8eaf0, --text-muted #7a7f96, --user-bubble linear-gradient(135deg,#6c63ff,#9b6dff), --bot-bubble rgba(255,255,255,0.05), --bot-border rgba(108,99,255,0.25), --danger #ff5a5f, --success #4ade80, --cached #f59e0b, --radius-lg 16px, --radius-md 12px, --radius-sm 8px, --transition 0.2s ease.
SOFTEN: --accent-glow → rgba(108,99,255,0.22). ADD: --radius-xl 20px, --focus-ring 0 0 0 3px rgba(108,99,255,0.28). Reduce glow shadows on user bubbles/avatar/send button to subtle ones (e.g. 0 2px 10px).

Required styles:
1. Shell: full-height app, not a small centered card — #chat-container { width: min(920px,100vw); height: 100dvh; margin: 0 auto; }. Body = dark bg, Inter, no noisy multi-radial ::before (at most one subtle gradient). ≤480px: full-bleed (no border/radius), composer padding-bottom: env(safe-area-inset-bottom), suggestion grid → 1 column.
2. #messages: flex column, gap 14px, overflow-y auto, scroll-behavior smooth, scrollbar-gutter stable, 4px thin scrollbar.
3. #chat-header: flex, subtle glass bg, bottom border; .logo (36px circle with inline-SVG graduation cap, no emoji); .info h1 (16px/700) + p (12px muted); #status-dot restyled as a STATUS PILL (green pulsing dot + text) pushed right (margin-left auto); #clear-btn icon button, muted, hover → danger, focus ring.
4. .message: flex, max-width 88%, entrance animation (fade+8px up). .user align-self flex-end (row-reverse), .bot flex-start. Bot bubbles: .avatar (28px circle) + .bubble.
5. .bubble: padding 12px 16px, radius --radius-lg with a 4px tail corner on the sender's bottom corner, font 14.5px/1.65, word-break break-word. .message.user .bubble = gradient, white text, subtle shadow. .message.bot .bubble = --bot-bubble + faint border, NO heavy glow.
6. .msg-time: 10.5px --text-muted, margin-top 6px, block, right-aligned for user / left for bot.
7. .msg-actions: flex row gap 6px, margin-top 8px; .action-btn transparent muted 11px icon+label, hover → accent. Hidden until .message:hover/:focus-within on hover-capable devices; ALWAYS visible under @media (hover: none).
8. .message.error .bubble: bg rgba(255,90,95,0.10), border rgba(255,90,95,0.35), text #ff8c8f.
9. .badge-row (flex, gap 6px, wrap, margin-top 8px) containing .cached-badge (amber chip: ⚡ Cached response) and .response-time-badge (muted chip: ⏱ Responded in X.XXs) — keep both classes, restyle quieter (11px, 6px radius, tinted bg).
10. .source-link: keep as a quiet pill/link, hover accent.
11. #typing-indicator: hidden by default; .bubble contains #skeleton = 3 .sk-bar shimmer bars (height 10px; widths 220/170/120px; radius 6px; gradient shimmer keyframes 1.4s infinite) — replaces the old 3 bouncing dots (the .dot markup is gone).
12. #prompt-chips: CSS grid repeat(2,1fr) gap 10px; .chip/.suggestion-card = card (bg --bg-glass, 1px --border, radius --radius-md, padding 12px 14px, text-left, cursor pointer), hover lifts 1px + accent border, focus-visible ring, contains an emoji label + muted .hint line.
13. #input-area: bordered bar, flex, align-items flex-end; #query-input pill (border-radius 24px, padding 11px 16px, resize none, max-height 120px, focus = --focus-ring); #send-btn 44px CIRCLE (border-radius 50%), gradient bg, no heavy glow, disabled opacity 0.4; #send-btn.loading (or body.loading) swaps to a CSS spinning ring with spinner keyframes.
14. #char-count: 11px muted, right-aligned, min-height so layout never jumps; .warn → #f59e0b; .full → --danger.
15. #error-toast: quiet red-tinted toast, absolute, bottom above composer, opacity/transform transition, .show visible.
16. Focus & motion: :focus-visible { box-shadow: var(--focus-ring); } for all interactive elements; @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

## PART B — HTML BODY (markup)
<div id="chat-container" role="main"> contains:
1. #chat-header: .logo (inline SVG graduation cap), .info > h1 "WCE Bot" + p "Walchand College of Engineering · Official Information Assistant", #status-dot pill (small green pulsing dot + <span>Online</span>, title="Service online", aria-label="Service online"), #clear-btn <button type="button" onclick="clearChat()" aria-label="Clear conversation" title="Clear chat"> with inline SVG trash icon.
2. #messages (role="log" aria-label="Chat messages" aria-live="polite") containing ONE child: <div class="message bot" id="welcome-msg"><div class="bubble"> greeting — short, 2 lines, e.g. 👋 Hello! I'm <strong>WCE Bot</strong> — ask me about fees, syllabus, admissions or exams, all from official college documents. <em>What would you like to know?</em> </div></div>. No long <br> walls. JS appends everything else.
3. <div id="typing-indicator" class="message bot" role="status" aria-label="WCE Bot is typing"><div class="bubble"><div id="skeleton"><span class="sk-bar"></span><span class="sk-bar"></span><span class="sk-bar"></span></div></div></div>  ← keep this exact id (JS toggles its display).
4. #prompt-chips (aria-label="Quick question suggestions") with FOUR <button type="button" class="chip suggestion-card" onclick="sendChip(this)"> of ids chip-1..chip-4. Text MUST start with emoji + space (sendChip strips it):
   chip-1 💰 Fee structure (+ hint "Tuition, hostel, exam fees")
   chip-2 📚 Syllabus for CS (+ hint "Semester-wise course plan")
   chip-3 📅 Admission deadlines (+ hint "Dates, eligibility, quota")
   chip-4 📄 Download prospectus (+ hint "PDF overview of the college")
5. #input-area: <textarea id="query-input" placeholder="Ask about fees, syllabus, admissions…" rows="1" maxlength="500" aria-label="Type your question" onkeydown="handleKeydown(event)" oninput="autoResize(this)"></textarea> + <button id="send-btn" type="button" onclick="sendMessage()" aria-label="Send message" title="Send (Enter)"> inline SVG paper-plane (line + polygon) </button> + <div id="char-count">0/500</div>.
6. <div id="error-toast" role="alert" aria-live="assertive"></div>
</div><!-- /chat-container -->
Script tag at end of body.

## PART C — JavaScript (<script>, vanilla, global scope)
State/refs: const API_BASE = "http://localhost:8000"; let isLoading = false; let lastQuery = ""; const messagesEl/inputEl/sendBtn/typingEl(#typing-indicator)/chipsEl(#prompt-chips)/errorToastEl(#error-toast)/charCountEl(#char-count) = getElementById(...). Also cache const welcomeTpl = the #welcome-msg cloneNode(true) at load.

KEEP THESE BEHAVIORS EXACTLY (reuse the logic):
- autoResize(el): height auto → min(scrollHeight,120)+"px"; ALSO calls updateCount().
- handleKeydown(e): Enter && !shiftKey → e.preventDefault(); sendMessage().
- sendChip(el): text = el.textContent.replace(/^[^\w]+/,"").trim(); inputEl.value = text; sendMessage().
- escapeHtml(str) / escapeAttr(str): full entity escaping (&, <, >, ", ').
- scrollToBottom(): requestAnimationFrame → messagesEl.scrollTop = messagesEl.scrollHeight.
- showError(msg): textContent, add .show, setTimeout remove .show after 4000.
- setLoading(state): isLoading = state; sendBtn.disabled = state; inputEl.disabled = state; typingEl.style.display = state ? "flex" : "none"; sendBtn.classList.toggle("loading", state); state ? messagesEl.setAttribute("aria-busy","true") : messagesEl.removeAttribute("aria-busy"); if (state) scrollToBottom().
- appendMessage(role, html, raw=false): .message.{role} > .bubble; raw ? innerHTML : textContent. EXTENDED: every bubble also gets <span class="msg-time"> with now() (HH:MM, 24h, zero-padded) appended after the content.
- sendMessage(): if (isLoading) return; const query = inputEl.value.trim(); if (!query) return; lastQuery = query; chipsEl.style.display = "none"; appendMessage("user", escapeHtml(query)); inputEl.value = ""; inputEl.style.height = "auto"; updateCount();
  try { const t0 = performance.now(); const response = await fetch(`${API_BASE}/chat`, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({query})}); const data = await response.json(); const elapsedMs = performance.now() - t0;
    if (!response.ok) { const detail = data.detail || `Error ${response.status}`; if (response.status === 429) showError("⏱ Rate limit reached — please wait a moment before asking again."); else if (response.status === 400) showError("⚠ " + detail); else showError("Something went wrong. Please try again."); appendErrorBubble(query); return; }
    renderBotAnswer(data, elapsedMs);
  } catch (err) { console.error(err); showError("🔌 Could not reach WCE Bot. Make sure the backend is running."); appendErrorBubble(query); } finally { setLoading(false); }
- renderBotAnswer(data, elapsedMs): const {answer, source_url, cached} = data; let html = `<p>${escapeHtml(answer).replace(/\n/g,"<br>")}</p>`; if (source_url) html += `<a class="source-link" href="${escapeAttr(source_url)}" target="_blank" rel="noopener noreferrer">📄 View Document</a>`; if (cached || typeof elapsedMs === "number") { html += `<div class="badge-row">`; if (cached) html += `<span class="cached-badge">⚡ Cached response</span>`; if (typeof elapsedMs === "number") { const secs = elapsedMs/1000; const label = secs < 10 ? `${secs.toFixed(2)}s` : `${secs.toFixed(1)}s`; html += `<span class="response-time-badge">⏱ Responded in ${label}</span>`; } html += `</div>`; html += `<div class="msg-actions"><button type="button" class="action-btn" onclick="copyBubble(this)">⧉ Copy</button></div>`; appendMessage("bot", html, true); }

NEW FUNCTIONS (declare with `function` so inline onclick resolves):
- now(): Date → "HH:MM" zero-padded 24h.
- updateCount(): n = inputEl.value.length; charCountEl.textContent = n + "/500"; toggle "warn" (n>450 && n<500) and "full" (n>=500).
- clearChat(): if (isLoading) return; messagesEl.innerHTML = ""; messagesEl.appendChild(welcomeTpl.cloneNode(true)); chipsEl.style.display = ""; errorToastEl.classList.remove("show"); updateCount(); scrollToBottom().
- appendErrorBubble(query): builds <div class="message error bot"> with .bubble containing escaped text "⚠ Couldn't get an answer. Check that the backend is running, then try again." + .msg-time + .msg-actions row with [Retry] onclick="retrySend()" and [Copy] onclick="copyBubble(this)"; append via messagesEl.appendChild(wrapper) + scrollToBottom().
- retrySend(): if (isLoading) return; if (!lastQuery) return; inputEl.value = lastQuery; sendMessage().
- copyBubble(btn): const bubble = btn.closest(".message").querySelector(".bubble"); const clone = bubble.cloneNode(true); clone.querySelectorAll(".msg-actions").forEach(n => n.remove()); const text = clone.innerText.trim(); if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => flash(btn)).catch(() => showError("Copy failed")); else showError("Copy failed");
- flash(btn): save original label, set to "✓ Copied", setTimeout restore 1500ms.

## SELF-CHECK BEFORE OUTPUT (silently)
- CSS has rules for every class/id used in markup: chat-container, chat-header, logo, info, status-dot, clear-btn, messages, welcome-msg, message/user/bot/error, avatar, bubble, msg-time, msg-actions, action-btn, badge-row, cached-badge, response-time-badge, source-link, typing-indicator, skeleton, sk-bar, prompt-chips, chip, suggestion-card, hint, input-area, query-input, send-btn (incl. .loading), char-count (.warn/.full), error-toast (.show).
- Markup has ids: chat-container, chat-header, status-dot, clear-btn, messages, welcome-msg, typing-indicator, skeleton, prompt-chips, chip-1..chip-4, input-area, query-input, send-btn, char-count, error-toast.
- Script defines: autoResize, handleKeydown, sendChip, escapeHtml, escapeAttr, scrollToBottom, showError, setLoading, appendMessage, sendMessage, renderBotAnswer, now, updateCount, clearChat, appendErrorBubble, retrySend, copyBubble, flash.
- Balanced tags, no unclosed braces, no TODOs.

Output the complete file now, in a single ```html code block.
````

**Size:** ~900–1,100 lines ≈ 15–25k output tokens — fits inside Gemini 3.8 Flash's 64k output cap, but consumes a large slice of a free-plan weekly quota in one go.

---

## Strategy C — Three-chunk prompts (fallback)

Run **1 → 2 → 3**, pasting each result into `frontend/index.html` before sending the next.

### C1 — `<style>` block

````text
You are restyling the frontend of "WCE Bot", a dark-theme college chatbot. This is CHUNK 1 OF 3: output ONLY the contents of the <style> ... </style> block.

OUTPUT RULES (critical):
- Output exactly one ```html code block containing only the CSS between <style> and </style> tags (include the tags).
- NO explanations, NO "...rest", NO "// same as before", NO placeholders. Every rule complete.
- The file this CSS goes into is a single self-contained index.html using Google Font "Inter" (already linked). No frameworks, no preprocessors, no CDN CSS.

## Design tokens — REFINED DARK (keep dark, remove the heavy glow/gamer look)
Keep existing palette, adjust: --accent-glow softened to rgba(108,99,255,0.22); add --radius-xl: 20px; add --focus-ring: 0 0 0 3px rgba(108,99,255,0.28); add a 4px-based spacing scale. Reduce/soften glow shadows on user bubbles, avatar and send button to subtle ones (e.g. 0 2px 10px). Existing tokens to keep: --bg #0d0f14, --bg-card #13161e, --bg-glass, --border, --accent #6c63ff, --accent-dim, --text, --text-muted, --user-bubble gradient, --bot-bubble, --bot-border, --danger #ff5a5f, --success #4ade80, --cached #f59e0b, --radius-lg/md/sm, --transition.

## Required selectors (element IDs are fixed because JavaScript targets them — do NOT rename)
- #chat-container, #chat-header, #status-dot, #messages, #typing-indicator, #prompt-chips, .chip, #input-area, #query-input, #send-btn, #error-toast — all must exist and remain styled.
- #messages: flex column, overflow-y auto, scrollbar-gutter: stable, thin scrollbar (4px), gap 14px, smooth scroll.
- Body/HTML: dark bg, Inter font, height 100%.

## New selectors to design
1. Shell: full-height app, not a small centered card — width: min(920px, 100vw); height: 100dvh; margin: 0 auto; subtle 1px border. Drop the noisy multi-radial body::before (at most one very subtle gradient).
2. Status pill: #status-dot becomes a pill — green dot + text label, subtle border, aria-label unchanged (JS/HTML handles content).
3. #clear-btn: icon button in header, muted color, hover turns danger-red, focus-visible ring, disabled-safe.
4. .avatar: small circular SVG holder (36px) for bot messages; header .logo uses an inline SVG mark, no emoji.
5. .bubble: refined radii (--radius-lg top, small tail corner 4px on the sender's bottom corner), font-size 14.5px, line-height 1.65, subtle border, no heavy glow. .message.user .bubble uses --user-bubble gradient with reduced shadow; .message.bot .bubble uses --bot-bubble + faint border.
6. .msg-time: 10.5px, --text-muted, appears under bubble content, right-aligned for user / left for bot.
7. .msg-actions: row of .action-btn (Copy, Retry) under bot + error bubbles — transparent, muted, 11px, icon+label; hidden (opacity 0) on hover-capable devices until .message:hover / :focus-within, always visible on touch (@media (hover: none)).
8. .message.error .bubble: red-tinted background (rgba(255,90,95,0.10)), border rgba(255,90,95,0.35), text #ff8c8f.
9. #skeleton (inside #typing-indicator): 3 shimmer bars (bar heights ~10px, widths 220/170/120px), border-radius 6px, gradient shimmer keyframes 1.4s infinite; hide under prefers-reduced-motion.
10. .suggestion-card (replaces pill look of .chip): 2x2 CSS grid (#prompt-chips { display:grid; grid-template-columns: repeat(2,1fr); gap:10px }), card = bg var(--bg-glass), 1px --border, radius --radius-md, padding 12px 14px, left-aligned, hover lifts 1px + border-color accent, focus-visible ring; still clickable like .chip (style both .chip and .suggestion-card identically if needed).
11. #input-area: bordered bar with pill textarea — #query-input border-radius 24px, padding 11px 16px, same focus ring as --focus-ring; #send-btn becomes 44px circle (border-radius 50%), gradient background, subtle shadow, disabled opacity.
12. #send-btn.loading (or body.loading): the SVG swaps to a CSS spinning ring — spinner keyframes, animation disabled under prefers-reduced-motion.
13. #char-count: 11px, --text-muted, right-aligned under the composer; .warn (above 450 chars) → #f59e0b; .full (500) → --danger.
14. #error-toast: keep current 4s toast, restyle quieter, bottom offset above composer.
15. @media (max-width: 480px): full-bleed container (border: none), composer padding-bottom: env(safe-area-inset-bottom), cards single column.
16. @media (prefers-reduced-motion: reduce): disable all animations/transitions.
17. :focus-visible { outline: none; box-shadow: var(--focus-ring); } for buttons/cards.

Output the complete <style> block now.
````

### C2 — `<body>` markup

````text
You are rewriting the markup of "WCE Bot", a dark-theme college chatbot. This is CHUNK 2 OF 3: output ONLY the HTML from <body> to </body>.

OUTPUT RULES (critical):
- Exactly one ```html code block containing only <body> ... </body> (include the tags).
- NO explanations, NO commented-out old markup, NO "...rest", NO placeholders.
- No frameworks, no external assets except the already-linked Google Font Inter. UI icons = inline SVG only (emoji allowed ONLY inside suggestion-card labels). Keep <style> and <script> as siblings outside this output (they already exist in the file).

## Structure required (element IDs are FIXED — JavaScript does getElementById on every one of them)
<div id="chat-container" role="main">

1. HEADER — #chat-header:
   - .logo: inline SVG graduation-cap mark inside a circle (replaces the old 🎓 avatar).
   - .info > h1 "WCE Bot" + p "Walchand College of Engineering · Official Information Assistant".
   - #status-dot: redesign as a status PILL containing a small green pulsing dot + <span>Online</span>; keep title="Service online" and aria-label="Service online".
   - NEW button #clear-btn (type="button", onclick="clearChat()", aria-label="Clear conversation", title="Clear chat") with inline SVG trash icon.

2. MESSAGES — #messages (role="log" aria-label="Chat messages" aria-live="polite"):
   - Welcome bubble: <div class="message bot" id="welcome-msg"><div class="bubble"> ... </div></div>
     Content: short 2-line greeting (👋 Hello! I'm <strong>WCE Bot</strong> — ask me about fees, syllabus, admissions or exams, all sourced from official college documents. <em>What would you like to know?</em>). No long <br> walls.
   - Nothing else; messages are appended by JS.

3. LOADER — <div id="typing-indicator" class="message bot" role="status" aria-label="WCE Bot is typing"> containing .bubble > #skeleton > three <span class="sk-bar"></span> (widths 220/170/120px). Keep this element id exactly — JS toggles its display.

4. SUGGESTIONS — #prompt-chips (aria-label="Quick question suggestions"): FOUR buttons, class="chip suggestion-card", ids chip-1..chip-4, each onclick="sendChip(this)". Text must start with an EMOJI + space, then label (sendChip strips the emoji):
   chip-1: 💰 Fee structure
   chip-2: 📚 Syllabus for CS
   chip-3: 📅 Admission deadlines
   chip-4: 📄 Download prospectus
   (Each card may also include a small muted <span class="hint"> one-liner, e.g. "Tuition, hostel, exam fees" — but the button's leading text must remain the emoji + label.)

5. COMPOSER — #input-area:
   - <textarea id="query-input" placeholder="Ask about fees, syllabus, admissions…" rows="1" maxlength="500" aria-label="Type your question" onkeydown="handleKeydown(event)" oninput="autoResize(this)"></textarea>
   - <button id="send-btn" type="button" onclick="sendMessage()" aria-label="Send message" title="Send (Enter)"> with inline SVG paper-plane (keep the existing two-element SVG: line + polygon).
   - <div id="char-count" aria-live="off">0/500</div> (JS updates text and classes warn/full).

6. <div id="error-toast" role="alert" aria-live="assertive"></div>

Then </div><!-- /chat-container --> and </body>.

Copy the exact wording of existing texts (title, subtitle, placeholders, chips, aria-labels) unless told above. Output the complete <body> block now.
````

### C3 — `<script>` block

````text
You are updating the JavaScript of "WCE Bot", a college chatbot frontend. This is CHUNK 3 OF 3: output ONLY the <script> ... </script> block for a single-file index.html (vanilla JS, no frameworks).

OUTPUT RULES (critical):
- Exactly one ```javascript code block containing only <script> ... </script> (include the tags).
- NO explanations, NO "...rest of the file", NO omitted functions. Output EVERY function in full.
- The <style> and <body> already exist in the file; only the script changes.

## DOM ids available (all exist in the markup — getElementById them)
chat-container, chat-header, status-dot, clear-btn, messages, welcome-msg, typing-indicator (contains #skeleton), prompt-chips, chip-1..chip-4, input-area, query-input, send-btn, char-count, error-toast.

## KEEP THESE EXACT BEHAVIORS (copy the logic, do not rename)
const API_BASE = "http://localhost:8000";
let isLoading = false;
DOM refs: messagesEl, inputEl, sendBtn, typingEl (#typing-indicator), chipsEl (#prompt-chips), errorToastEl (#error-toast).

- autoResize(el): height auto → min(scrollHeight, 120)+"px".
- handleKeydown(e): Enter without Shift → preventDefault + sendMessage().
- sendChip(el): text = el.textContent.replace(/^[^\w]+/,"").trim(); inputEl.value = text; sendMessage(). (This is why suggestion cards start with an emoji.)
- escapeHtml(str) and escapeAttr(str): full entity escaping as before.
- scrollToBottom(): requestAnimationFrame → messagesEl.scrollTop = scrollHeight.
- showError(msg): set textContent, add .show, remove after 4000ms.
- setLoading(state): isLoading, sendBtn.disabled, inputEl.disabled, typingEl display flex/none, scrollToBottom on true, AND now also toggle a "loading" class on sendBtn (for the spinner) plus set/remove aria-busy on messagesEl.

- appendMessage(role, html, raw=false): builds .message.{role} > .bubble; raw ? innerHTML : textContent. EXTEND IT: signature appendMessage(role, html, raw=false) stays, but every bubble also gets a <span class="msg-time"> showing current time as HH:MM (24h, pad zeros) appended after the content (for raw HTML, append the span inside the bubble; for user text use textContent then append the span separately).

- sendMessage(): async. If isLoading return; query = inputEl.value.trim(); if (!query) return. Save lastQuery = query (module scope, used by retry). Hide chipsEl. appendMessage("user", escapeHtml(query)). Clear input (value="", height auto, reset char count). setLoading(true). try { const t0 = performance.now(); const response = await fetch(`${API_BASE}/chat`, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({query})}); const data = await response.json(); const elapsedMs = performance.now() - t0; if (!response.ok) { same 429/400/else toast messages as before via showError(...); then appendErrorBubble(query); return; } renderBotAnswer(data, elapsedMs); } catch (err) { console.error(err); showError("🔌 Could not reach WCE Bot. Make sure the backend is running."); appendErrorBubble(query); } finally { setLoading(false); }

- renderBotAnswer(data, elapsedMs): unchanged semantics — escape answer, \n → <br>, optional source-link (escapeAttr on href), then if (cached || typeof elapsedMs === "number") build <div class="badge-row"> with optional <span class="cached-badge">⚡ Cached response</span> and (when elapsedMs is a number) <span class="response-time-badge">⏱ Responded in {label}</span> where label = secs<10 ? secs.toFixed(2)+"s" : secs.toFixed(1)+"s". Then appendMessage("bot", html, true).

## NEW functions to write
- now(): returns "HH:MM" from a Date, zero-padded.
- updateCount(): const n = inputEl.value.length; charCountEl.textContent = n + "/500"; charCountEl.classList.toggle("warn", n > 450 && n < 500); charCountEl.classList.toggle("full", n >= 500).
- clearChat(): if isLoading return; messagesEl.innerHTML = ""; append a cached clone of the welcome bubble (const welcomeTpl = #welcome-msg cloneNode(true) captured at load); chipsEl.style.display = ""; errorToastEl.classList.remove("show"); updateCount(); scrollToBottom().
- appendErrorBubble(query): <div class="message error bot"> with .bubble containing escaped "⚠ Couldn't get an answer. Check that the backend is running, then try again.", a .msg-time, and .msg-actions with [Retry] onclick="retrySend()" and [Copy] onclick="copyBubble(this)". Store the failed query in lastQuery.
- retrySend(): if isLoading return; if (!lastQuery) return; inputEl.value = lastQuery; sendMessage().
- copyBubble(btn): btn.closest(".message").querySelector(".bubble") → clone it → remove .msg-actions from the clone → innerText → navigator.clipboard.writeText(text).then(() => flash(btn)).catch(() => showError("Copy failed")); fallback showError("Copy failed") if clipboard unavailable.
- flash(btn): save original label, set "✓ Copied", restore after 1500ms.
- clear-btn: markup already has onclick="clearChat()", so just DEFINE clearChat with `function` so the inline handler resolves.

## Global scope note
All functions must be declared with `function` (or assigned to window) so inline onclick="..." attributes resolve. No modules, no DOMContentLoaded wrapper (script is at end of body).

Output the complete <script> block now.
````

---

## Model notes (Gemini 3.8 Flash)

| Fact | Value |
|---|---|
| Max output tokens | 65,536 (includes thinking tokens) |
| Thinking levels | low / medium / high (`medium` default; `minimal` errors) |
| Context window | 1M tokens |
| Free tier (Antigravity) | weekly quota, exact figure unpublished, work-based |

Strategy A never approaches the output cap (edits ≤200 lines each). Strategy B needs ~15–25k of 64k. Strategy C needs ~5–8k per response.

## Verification (after any strategy)

1. `node --check` on the extracted `<script>` block
2. `pytest -q` → expect **117 passed** (backend must be untouched)
3. Open `frontend/index.html` via `file://` with the `uvicorn --reload` server running, then test: send message, timestamp shown, Copy, Retry, Clear chat, char counter (warn/full), skeleton loader, cached + response-time badges, mobile width
4. `git diff --stat` — only `frontend/index.html` should appear (no commit unless asked)
