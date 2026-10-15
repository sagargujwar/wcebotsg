#!/usr/bin/env python
"""
scripts/demo_proxy.py
─────────────────────
Local mirror of the college website with the floating WCE Bot widget injected.

    python scripts/demo_proxy.py
    → open http://localhost:9000 in a browser

What it does:
  1. Forwards every request to BASE_URL (from .env), path-for-path.
  2. On HTML pages: rewrites same-origin links to stay inside the proxy and
     injects <script src="<backend>/widget.js"> before </body> — the floating
     chat button then appears on the real college page, and clicking it opens
     the chat as an overlay on the same screen (no navigation).
  3. Everything else (CSS, JS, images, PDFs) is streamed through unchanged.

Why a proxy: https://walchandsangli.ac.in/ is a live site we don't control, so
this is the way to *demonstrate* the embed snippet without editing the site.
With site access, just paste the one-line snippet instead (see README).

Only stdlib http.server + requests/bs4 (already project dependencies).
"""
from __future__ import annotations

import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("demo_proxy")

_SETTINGS = get_settings()
_UPSTREAM = urlparse(_SETTINGS.base_url)
_ORIGIN = f"{_UPSTREAM.scheme}://{_UPSTREAM.netloc}"
_WIDGET_JS = _SETTINGS.widget_api_base.rstrip("/") + "/widget.js"
_CA_BUNDLE = _SETTINGS.ssl_ca_bundle.strip() or True
_UPSTREAM_TIMEOUT = 20
_USER_AGENT = "WCEBot-DemoProxy/1.0 (local widget demo)"

# Headers that must not pass through: they are origin-scoped and would either
# break the proxy (HSTS forcing https on localhost) or block the injected
# widget script (CSP).
_STRIP_HEADERS = {
    "content-security-policy",
    "content-security-policy-report-only",
    "strict-transport-security",
    "alt-svc",
    "x-frame-options",
    "content-length",  # recomputed after any rewrite
    "content-encoding",  # requests already decompressed the body
    "transfer-encoding",
    "connection",
}


def _rewrite_html(html: str) -> str:
    """Inject the widget snippet and keep same-origin links on the proxy."""
    soup = BeautifulSoup(html, "html.parser")

    origin = _ORIGIN
    netloc = _UPSTREAM.netloc
    for tag in soup.find_all(True):
        for attr in ("href", "src", "action", "poster", "data-src"):
            val = tag.get(attr)
            if not isinstance(val, str) or not val:
                continue
            if val.startswith(origin):
                path = val[len(origin):] or "/"
                tag[attr] = path
            elif val.startswith("//" + netloc):
                path = val[len(netloc) + 1:] or "/"
                tag[attr] = path

    # <base> would send relative URLs back to the live site
    for base in soup.find_all("base"):
        base.decompose()

    if soup.body:
        snippet = soup.new_tag("script", src=_WIDGET_JS, defer="")
        soup.body.append(snippet)
    else:
        snippet = soup.new_tag("script", src=_WIDGET_JS, defer="")
        soup.append(snippet)

    return str(soup)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WCEBotDemoProxy/1.0"

    # ── HTTP verbs ────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        self._proxy("GET")

    def do_HEAD(self) -> None:
        self._proxy("HEAD")

    def do_POST(self) -> None:
        self._proxy("POST")

    # ── Core ──────────────────────────────────────────────────────────────────

    def _proxy(self, method: str) -> None:
        upstream_url = _UPSTREAM.scheme + "://" + _UPSTREAM.netloc + self.path
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": self.headers.get("Accept", "*/*"),
            "Accept-Language": self.headers.get("Accept-Language", "en"),
            # Ask for identity so no decompression/re-encoding dance is needed
            "Accept-Encoding": "identity",
        }
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""

        try:
            resp = requests.request(
                method,
                upstream_url,
                headers=headers,
                data=body,
                timeout=_UPSTREAM_TIMEOUT,
                allow_redirects=True,
                verify=_CA_BUNDLE,
            )
        except requests.RequestException as exc:
            logger.warning("Upstream failed for %s: %s", upstream_url, exc)
            self._send_error_page(str(exc))
            return

        content_type = resp.headers.get("Content-Type", "")
        is_html = "html" in content_type.lower()

        if is_html and method != "HEAD":
            try:
                payload = _rewrite_html(resp.text).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            except Exception as exc:
                logger.warning("HTML rewrite failed for %s: %s", upstream_url, exc)
                payload = resp.content
        else:
            payload = resp.content

        self.send_response(resp.status_code)
        for name, value in resp.headers.items():
            if name.lower() in _STRIP_HEADERS:
                continue
            self.send_header(name, value)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(payload)
        logger.info("%s %s -> %d (%d bytes)", method, self.path, resp.status_code, len(payload))

    def _send_error_page(self, detail: str) -> None:
        page = (
            "<!doctype html><meta charset='utf-8'><title>WCE Bot demo proxy</title>"
            "<body style='font-family:sans-serif;padding:40px'>"
            "<h1>Upstream unavailable</h1>"
            f"<p>The college site could not be reached: <code>{detail}</code></p>"
            "<p>Is the network up? Try again, or check <code>BASE_URL</code> in .env.</p>"
            "</body>"
        ).encode("utf-8")
        self.send_response(502)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, fmt: str, *args) -> None:
        logger.debug(fmt, *args)


def main() -> None:
    port = _SETTINGS.demo_proxy_port
    if not _SETTINGS.base_url.startswith(("http://", "https://")):
        logger.error("BASE_URL must be an http(s) URL — got %r", _SETTINGS.base_url)
        sys.exit(1)

    logger.info("Mirroring %s", _ORIGIN)
    logger.info("Widget script : %s", _WIDGET_JS)
    logger.info("Open          : http://localhost:%d", port)

    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping demo proxy.")
        server.server_close()


if __name__ == "__main__":
    main()
