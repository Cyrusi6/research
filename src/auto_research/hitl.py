"""Human-in-the-loop with Feishu integration.

The pipeline pauses at decision gates, sends a Feishu message to the user,
and waits for guidance before proceeding.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .utils import now_utc


@dataclass
class HITLDecision:
    action: str  # "approve" | "reject" | "guide"
    guidance: str = ""
    responded_at: str = ""


class FeishuClient:
    """Minimal Feishu (Lark) API client for sending messages and polling replies."""

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(self, config: dict[str, Any]):
        feishu_cfg = config.get("hitl", {}).get("feishu", {})
        self.app_id = feishu_cfg.get("app_id", "")
        self.app_secret = feishu_cfg.get("app_secret", "")
        self.user_open_id = feishu_cfg.get("user_open_id", "")
        self.user_email = feishu_cfg.get("user_email", "")
        self._token: str = ""
        self._token_expires_at: float = 0

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        url = f"{self.BASE_URL}/auth/v3/tenant_access_token/internal"
        resp = requests.post(
            url,
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {data}")
        self._token = data["tenant_access_token"]
        self._token_expires_at = time.time() + data.get("expire", 3600)
        return self._token

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.app_secret and (self.user_open_id or self.user_email))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }

    # ------------------------------------------------------------------
    # User lookup
    # ------------------------------------------------------------------

    def resolve_open_id(self) -> str:
        if self.user_open_id:
            return self.user_open_id
        if self.user_email:
            url = f"{self.BASE_URL}/contact/v3/users"
            resp = requests.get(
                url,
                headers=self._headers(),
                params={"user_id_type": "open_id", "page_size": 50},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Feishu user lookup failed: {data}")
            for item in data.get("data", {}).get("items", []):
                if item.get("email") == self.user_email:
                    self.user_open_id = item["open_id"]
                    return self.user_open_id
        raise RuntimeError("Cannot resolve Feishu user open_id")

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send_card(self, title: str, sections: list[dict[str, Any]], *, open_id: str | None = None) -> str:
        """Send an interactive card message. Returns message_id."""
        receive_id = open_id or self.resolve_open_id()
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [],
        }
        for sec in sections:
            if sec.get("type") == "markdown":
                card["elements"].append({
                    "tag": "markdown",
                    "content": sec["content"],
                })
            elif sec.get("type") == "divider":
                card["elements"].append({"tag": "hr"})
            elif sec.get("type") == "note":
                card["elements"].append({
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": sec["content"]}],
                })

        url = f"{self.BASE_URL}/im/v1/messages?receive_id_type=open_id"
        body = {
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        resp = requests.post(url, headers=self._headers(), json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu send message failed: {data}")
        return data["data"]["message_id"]

    def send_text(self, text: str, *, open_id: str | None = None) -> str:
        """Send a plain text message. Returns message_id."""
        receive_id = open_id or self.resolve_open_id()
        url = f"{self.BASE_URL}/im/v1/messages?receive_id_type=open_id"
        body = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        resp = requests.post(url, headers=self._headers(), json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu send text failed: {data}")
        return data["data"]["message_id"]

    # ------------------------------------------------------------------
    # Receive / poll
    # ------------------------------------------------------------------

    def get_message(self, message_id: str) -> dict[str, Any]:
        url = f"{self.BASE_URL}/im/v1/messages/{message_id}"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu get message failed: {data}")
        return data["data"]["items"][0]

    def poll_user_reply(
        self,
        sent_message_id: str,
        *,
        timeout_seconds: int = 3600,
        poll_interval: int = 10,
    ) -> str | None:
        """Poll for a user reply after sending a message. Returns reply text or None on timeout.

        Uses the chat_id from the sent message to list recent messages
        and find any user reply that arrived after our message.
        """
        msg_info = self.get_message(sent_message_id)
        chat_id = msg_info["chat_id"] or msg_info["thread_id"] or msg_info.get("parent_id")
        if not chat_id:
            chat_id = msg_info["root_id"]

        deadline = time.time() + timeout_seconds
        our_msg_id = msg_info["message_id"]

        while time.time() < deadline:
            url = f"{self.BASE_URL}/im/v1/messages"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    params={
                        "container_id_type": "chat",
                        "container_id": chat_id,
                        "page_size": 20,
                        "sort_type": "ByCreateTimeDesc",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 0:
                    time.sleep(poll_interval)
                    continue
                items = data.get("data", {}).get("items", [])
                for item in items:
                    if item.get("message_id") == our_msg_id:
                        continue
                    if item.get("sender", {}).get("id_type") == "app":
                        continue
                    if item.get("msg_type") == "text":
                        content_str = item.get("body", {}).get("content", "{}")
                        try:
                            content = json.loads(content_str)
                            return content.get("text", "")
                        except json.JSONDecodeError:
                            return content_str
            except Exception:
                pass
            time.sleep(poll_interval)

        return None


# ------------------------------------------------------------------
# HITL Manager
# ------------------------------------------------------------------


@dataclass
class HITLState:
    decision_file: Path
    decided: bool = False
    action: str = ""
    guidance: str = ""
    message_id: str = ""
    notified_at: str = ""


class HITLManager:
    """Orchestrates human-in-the-loop approvals via Feishu + local decision files."""

    def __init__(self, project_root: Path, config: dict[str, Any]):
        self.project_root = project_root
        self.config = config
        self.feishu = FeishuClient(config)
        self._state: HITLState | None = None

    @property
    def enabled(self) -> bool:
        return not self.config.get("orchestration", {}).get("auto_mode", True)

    def _decision_file(self) -> Path:
        return self.project_root / "meta" / "hitl_decision.json"

    # ------------------------------------------------------------------
    # Request approval
    # ------------------------------------------------------------------

    def request_approval(
        self,
        stage_key: str,
        stage_label: str,
        summary: str,
        *,
        artifacts: list[str] | None = None,
        blocking: bool = True,
        timeout_minutes: int = 60,
    ) -> HITLDecision:
        """Send Feishu notification and optionally block for user decision.

        In non-blocking mode, returns immediately with action="pending".
        The user can later run `auto-research decide` to unblock.
        """
        decision_file = self._decision_file()
        # Clear any previous decision
        decision_file.unlink(missing_ok=True)

        self._state = HITLState(decision_file=decision_file)

        if self.feishu.configured:
            self._send_feishu_notification(stage_key, stage_label, summary, artifacts)

        if not blocking:
            self._state.notified_at = now_utc()
            return HITLDecision(action="pending", guidance="Awaiting human decision via Feishu or CLI.")

        return self._wait_for_decision(stage_key, timeout_minutes=timeout_minutes)

    def _send_feishu_notification(
        self,
        stage_key: str,
        stage_label: str,
        summary: str,
        artifacts: list[str] | None,
    ) -> None:
        project_id = self.project_root.name
        registry_path = self.project_root / "meta" / "registry.yaml"
        import yaml

        if registry_path.exists():
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            topic = registry.get("research_topic", project_id)
        else:
            topic = project_id

        title = f"Auto-Research: {stage_key} 需要审批"
        sections = [
            {"type": "markdown", "content": f"**项目**: {project_id}\n**主题**: {topic}\n**阶段**: {stage_key} ({stage_label})"},
            {"type": "divider", "content": ""},
            {"type": "markdown", "content": f"**阶段摘要**\n{summary}"},
        ]
        if artifacts:
            artifact_lines = "\n".join(f"  • {a}" for a in artifacts[:10])
            sections.append({"type": "markdown", "content": f"**产物**\n{artifact_lines}"})

        sections.append({"type": "divider", "content": ""})
        sections.append({
            "type": "markdown",
            "content": (
                "**回复指令**\n"
                "• 回复 `approve` 批准并继续\n"
                "• 回复 `reject` 拒绝并回退\n"
                "• 回复 `guide: <你的指导意见>` 给出修改方向\n\n"
                f"或运行 CLI: `auto-research decide --project-id {project_id} --action <approve|reject|guide> [--guidance \"...\"]`"
            ),
        })
        sections.append({"type": "note", "content": f"⏰ {now_utc()}"})

        try:
            msg_id = self.feishu.send_card(title, sections)
            self._state.message_id = msg_id
            self._state.notified_at = now_utc()
        except Exception as e:
            # Feishu send failed – write to decision file so CLI can still work
            self._state.notified_at = now_utc()
            self._write_pending_decision(str(e))

    def _write_pending_decision(self, error_note: str = "") -> None:
        payload = {
            "status": "pending",
            "action": "",
            "guidance": "",
            "notified_at": self._state.notified_at if self._state else now_utc(),
            "feishu_error": error_note,
        }
        self._state.decision_file.parent.mkdir(parents=True, exist_ok=True)
        self._state.decision_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Wait for decision
    # ------------------------------------------------------------------

    def _wait_for_decision(self, stage_key: str, timeout_minutes: int = 60) -> HITLDecision:
        """Block until user provides a decision via CLI or Feishu reply."""
        deadline = time.time() + timeout_minutes * 60
        poll_interval = 10

        print(f"\n{'='*60}")
        print(f"[HITL] Stage {stage_key} awaiting human decision.")
        print(f"[HITL] Timeout: {timeout_minutes} minutes.")
        print(f"[HITL] Run: auto-research decide --project-id {self.project_root.name} --action <approve|reject|guide>")
        print(f"{'='*60}\n")

        while time.time() < deadline:
            # 1. Check local decision file (CLI-driven)
            decision = self._check_decision_file()
            if decision:
                return decision

            # 2. Poll Feishu for reply
            if self._state and self._state.message_id and self.feishu.configured:
                reply = self.feishu.poll_user_reply(
                    self._state.message_id,
                    timeout_seconds=5,
                    poll_interval=5,
                )
                if reply:
                    decision = self._parse_reply(reply)
                    self._persist_decision(decision)
                    return decision

            time.sleep(poll_interval)

        return HITLDecision(action="timeout", guidance="Human did not respond within timeout.")

    def _check_decision_file(self) -> HITLDecision | None:
        df = self._decision_file()
        if not df.exists():
            return None
        try:
            data = json.loads(df.read_text(encoding="utf-8"))
            if data.get("status") == "resolved":
                return HITLDecision(
                    action=data.get("action", "approve"),
                    guidance=data.get("guidance", ""),
                    responded_at=data.get("responded_at", ""),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def _parse_reply(self, text: str) -> HITLDecision:
        text = text.strip().lower()
        if text.startswith("reject"):
            return HITLDecision(action="reject", guidance=text, responded_at=now_utc())
        if text.startswith("approve"):
            return HITLDecision(action="approve", guidance="", responded_at=now_utc())
        if text.startswith("guide"):
            guidance = text.split(":", 1)[-1].strip() if ":" in text else text[5:].strip()
            return HITLDecision(action="guide", guidance=guidance, responded_at=now_utc())
        return HITLDecision(action="guide", guidance=text, responded_at=now_utc())

    def _persist_decision(self, decision: HITLDecision) -> None:
        df = self._decision_file()
        df.parent.mkdir(parents=True, exist_ok=True)
        df.write_text(
            json.dumps({
                "status": "resolved",
                "action": decision.action,
                "guidance": decision.guidance,
                "responded_at": decision.responded_at,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # CLI-triggered decision
    # ------------------------------------------------------------------

    def submit_decision(self, action: str, guidance: str = "") -> HITLDecision:
        decision = HITLDecision(action=action, guidance=guidance, responded_at=now_utc())
        self._persist_decision(decision)
        return decision
