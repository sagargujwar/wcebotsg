/**
 * frontend/widget.js — WCE Bot floating chat widget (embed loader).
 *
 * Host pages include it with one line:
 *
 *     <script src="http://localhost:8000/widget.js" defer></script>
 *
 * Optional override when the backend is hosted elsewhere:
 *
 *     <script src="https://cdn.example.com/widget.js" data-api="http://localhost:8000" defer></script>
 *
 * It injects a floating action button (bottom-right). Clicking it opens the
 * chat panel as an overlay on the SAME page — no navigation, no popup.
 * The panel is an <iframe> pointing at GET /widget on the backend, so host
 * page CSS can't break the chat UI and /chat is same-origin for the iframe.
 *
 * All elements carry a "wcebot-" prefix and use inline styles so the host
 * page's stylesheet is unlikely to interfere.
 */
(function () {
  "use strict";

  if (window.__WCEBOT_WIDGET__) return; // idempotent — load only once
  window.__WCEBOT_WIDGET__ = true;

  // ── API origin ─────────────────────────────────────────────────────────────
  var scriptEl = document.currentScript;
  var API_BASE = "http://localhost:8000";
  if (scriptEl) {
    if (scriptEl.getAttribute("data-api")) {
      API_BASE = scriptEl.getAttribute("data-api").replace(/\/+$/, "");
    } else if (scriptEl.src) {
      try {
        API_BASE = new URL(scriptEl.src).origin;
      } catch (e) { /* keep default */ }
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function make(tag, styles, attrs) {
    var node = document.createElement(tag);
    if (styles) {
      for (var k in styles) node.style[k] = styles[k];
    }
    if (attrs) {
      for (var a in attrs) {
        if (a === "text") node.textContent = attrs[a];
        else node.setAttribute(a, attrs[a]);
      }
    }
    return node;
  }

  var CHAT_ICON =
    '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>' +
    "</svg>";

  var ROOT_Z = "2147483646";

  // ── Floating action button ────────────────────────────────────────────────
  var fab = make("button", {
    position: "fixed",
    right: "24px",
    bottom: "24px",
    width: "60px",
    height: "60px",
    borderRadius: "50%",
    border: "none",
    background: "linear-gradient(135deg, #6c63ff, #9b6dff)",
    color: "#fff",
    boxShadow: "0 8px 24px rgba(108,99,255,0.45)",
    cursor: "pointer",
    zIndex: ROOT_Z,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "0",
    lineHeight: "0",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
  }, {
    type: "button",
    id: "wcebot-fab",
    "aria-label": "Open WCE Bot chat",
    "aria-expanded": "false",
    "aria-controls": "wcebot-panel",
    title: "Chat with WCE Bot",
  });
  fab.innerHTML = CHAT_ICON;
  fab.addEventListener("mouseenter", function () {
    fab.style.transform = "scale(1.06)";
  });
  fab.addEventListener("mouseleave", function () {
    fab.style.transform = "scale(1)";
  });

  // ── Chat panel (iframe overlay — same screen, no navigation) ──────────────
  var panel = make("div", {
    position: "fixed",
    right: "24px",
    bottom: "96px",
    width: "min(400px, calc(100vw - 32px))",
    height: "min(640px, calc(100vh - 140px))",
    borderRadius: "16px",
    overflow: "visible",
    background: "transparent",
    boxShadow: "0 16px 48px rgba(0,0,0,0.45)",
    zIndex: ROOT_Z,
    display: "none",
    opacity: "0",
    transform: "translateY(10px)",
    transition: "opacity 0.18s ease, transform 0.18s ease",
    border: "1px solid rgba(255,255,255,0.10)",
  }, { id: "wcebot-panel", role: "dialog", "aria-label": "WCE Bot chat" });

  var iframe = make("iframe", {
    width: "100%",
    height: "100%",
    border: "0",
    borderRadius: "16px",
    overflow: "hidden",
    background: "#13161e",
    display: "block",
  }, {
    src: API_BASE + "/widget",
    title: "WCE Bot — college information assistant",
    allow: "clipboard-write",
  });
  panel.appendChild(iframe);

  var closeBtn = make("button", {
    position: "absolute",
    top: "-10px",
    right: "-10px",
    width: "28px",
    height: "28px",
    borderRadius: "50%",
    border: "2px solid #13161e",
    background: "#1f2430",
    color: "#e8eaf0",
    cursor: "pointer",
    zIndex: "1",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "0",
    lineHeight: "1",
    fontSize: "16px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
  }, {
    type: "button",
    id: "wcebot-close",
    "aria-label": "Close WCE Bot chat",
    title: "Close chat",
    text: "\u00d7",
  });
  panel.appendChild(closeBtn); // inside the panel → positioned at its top-right

  // ── Toggle logic ──────────────────────────────────────────────────────────
  var isOpen = false;

  function openPanel() {
    isOpen = true;
    panel.style.display = "block";
    fab.setAttribute("aria-expanded", "true");
    fab.style.display = "none"; // close button takes over while open
    // let the browser paint the hidden state first, then animate in
    setTimeout(function () {
      if (!isOpen) return;
      panel.style.opacity = "1";
      panel.style.transform = "translateY(0)";
    }, 20);
  }

  function closePanel() {
    isOpen = false;
    panel.style.opacity = "0";
    panel.style.transform = "translateY(10px)";
    fab.setAttribute("aria-expanded", "false");
    fab.style.display = "flex";
    setTimeout(function () {
      if (!isOpen) panel.style.display = "none";
    }, 200);
  }

  fab.addEventListener("click", function () {
    isOpen ? closePanel() : openPanel();
  });
  closeBtn.addEventListener("click", closePanel);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && isOpen) closePanel();
  });

  // ── Mount (wait for <body>) ───────────────────────────────────────────────
  function mount() {
    document.body.appendChild(panel);
    document.body.appendChild(fab);
  }
  if (document.body) {
    mount();
  } else {
    document.addEventListener("DOMContentLoaded", mount);
  }
})();
