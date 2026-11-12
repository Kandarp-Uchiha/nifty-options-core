"""
Telegram alerts for the live trading loop.

Thin, dependency-free wrapper around the Bot API. Used for:
  - HALT engaged / cleared
  - Spread opened / closed
  - Phase 1 risk-check failures
  - End-of-day summary

Credentials live in config/secrets.env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

This module NEVER raises on send failure - alerts must not break the
trading loop. Failures are logged and silently dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib import parse, request

logger = logging.getLogger(__name__)


def _load_secrets(path: Path) -> dict[str, str]:
    """Tiny .env parser (no dependency on python-dotenv)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = v
    return out


class TelegramNotifier:
    """Sends one-off messages to a configured Telegram chat.

    Construct once at startup; reuse for the lifetime of the trading loop.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        *,
        secrets_path: Path | None = None,
        prefix: str = "",
        timeout_s: int = 10,
        enabled: bool = True,
    ) -> None:
        env: dict[str, str] = {}
        if secrets_path is not None:
            env = _load_secrets(secrets_path)
        self.token = (token or env.get("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (chat_id or env.get("TELEGRAM_CHAT_ID") or "").strip()
        self.prefix = prefix
        self.timeout_s = timeout_s
        self.enabled = enabled and bool(self.token) and bool(self.chat_id)
        if not self.enabled:
            logger.info("TelegramNotifier disabled (missing creds or enabled=False)")

    def send(self, text: str) -> bool:
        """Send a single text message. Returns True on success, False otherwise.
        NEVER raises."""
        if not self.enabled:
            return False
        body = f"{self.prefix}{text}" if self.prefix else text
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = parse.urlencode({
            "chat_id": self.chat_id,
            "text": body[:3900],
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = request.Request(url, data=payload, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                ok = resp.status == 200
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False
        if not ok:
            logger.error("Telegram send returned status %s", resp.status)
        return ok

    # ---------- convenience helpers ----------

    def halt_engaged(self, reason: str, source: str) -> bool:
        return self.send(f"HALT ENGAGED [{source}]\n{reason}")

    def halt_cleared(self, reason: str) -> bool:
        return self.send(f"HALT cleared\n{reason}")

    def spread_opened(self, spread_group_id: str, debit_inr: float, lots: int) -> bool:
        return self.send(
            f"Opened {spread_group_id}\n  lots={lots}  debit=INR {debit_inr:.0f}"
        )

    def spread_closed(self, spread_group_id: str, realised_inr: float, reason: str) -> bool:
        sign = "+" if realised_inr >= 0 else ""
        return self.send(
            f"Closed {spread_group_id}\n  pnl={sign}INR {realised_inr:.0f}  reason={reason}"
        )

    def eod_summary(
        self,
        *,
        day: str,
        n_closed: int,
        realised_inr: float,
        n_open: int,
        unrealised_inr: float,
        loss_streak: int,
    ) -> bool:
        sign_r = "+" if realised_inr >= 0 else ""
        sign_u = "+" if unrealised_inr >= 0 else ""
        return self.send(
            f"EOD {day}\n"
            f"  closed={n_closed}  realised={sign_r}INR {realised_inr:.0f}\n"
            f"  open={n_open}      unrealised={sign_u}INR {unrealised_inr:.0f}\n"
            f"  loss_streak={loss_streak}"
        )

    def risk_alert(self, condition: str, reason: str) -> bool:
        return self.send(f"RISK [{condition}]\n{reason}")
