# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import gzip
import json
import math
import os
import time
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import quote

import requests


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # numpy / pandas scalar compatibility
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    return str(obj)


class CloudStateStore:
    """
    GitHub Private Repository persistence.

    Required environment variable:
        GITHUB_STATE_TOKEN

    Optional environment variables:
        GITHUB_STATE_OWNER   default: xuz777-sudo
        GITHUB_STATE_REPO    default: crypto-strategy-state
        GITHUB_STATE_BRANCH  default: main
        GITHUB_STATE_PREFIX  default: state

    The Fine-grained PAT only needs:
        Repository access: crypto-strategy-state only
        Contents: Read and write
        Metadata: Read-only (automatic)
    """

    API_ROOT = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: int = 30,
    ):
        self.token = token or os.getenv("GITHUB_STATE_TOKEN", "").strip()
        self.owner = owner or os.getenv("GITHUB_STATE_OWNER", "xuz777-sudo").strip()
        self.repo = repo or os.getenv("GITHUB_STATE_REPO", "crypto-strategy-state").strip()
        self.branch = branch or os.getenv("GITHUB_STATE_BRANCH", "main").strip()
        self.prefix = (prefix or os.getenv("GITHUB_STATE_PREFIX", "state")).strip("/")
        self.timeout = int(timeout)

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "crypto-strategy-web-state/0.5",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    @property
    def configured(self) -> bool:
        return bool(self.token and self.owner and self.repo and self.branch)

    @property
    def available(self) -> bool:
        return self.configured

    def _repo_url(self) -> str:
        return f"{self.API_ROOT}/repos/{quote(self.owner)}/{quote(self.repo)}"

    def _object_path(self, key: str) -> str:
        key = str(key).strip("/")
        return f"{self.prefix}/{key}.json.gz"

    def _content_url(self, key: str) -> str:
        path = quote(self._object_path(key), safe="/")
        return f"{self._repo_url()}/contents/{path}"

    def status(self) -> dict:
        if not self.token:
            return {
                "ok": False,
                "configured": False,
                "message": "尚未設定 GITHUB_STATE_TOKEN",
            }

        try:
            response = self.session.get(
                self._repo_url(),
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return {
                    "ok": True,
                    "configured": True,
                    "message": f"GitHub 私有儲存已連線：{self.owner}/{self.repo}",
                }

            return {
                "ok": False,
                "configured": True,
                "message": (
                    f"GitHub 連線失敗 HTTP {response.status_code}："
                    f"{self._error_text(response)}"
                ),
            }
        except Exception as exc:
            return {
                "ok": False,
                "configured": True,
                "message": f"GitHub 連線失敗：{exc}",
            }

    @staticmethod
    def _error_text(response) -> str:
        try:
            data = response.json()
            return str(data.get("message", response.text))[:300]
        except Exception:
            return str(response.text)[:300]

    def _get_file_meta(self, key: str) -> Optional[dict]:
        response = self.session.get(
            self._content_url(key),
            params={"ref": self.branch},
            timeout=self.timeout,
        )

        if response.status_code == 404:
            return None

        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub GET 失敗 HTTP {response.status_code}: "
                f"{self._error_text(response)}"
            )

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("GitHub Contents API 回傳格式不正確")
        return data

    def save(self, key: str, payload: dict) -> dict:
        if not self.configured:
            raise RuntimeError("GitHub 狀態儲存尚未完成環境變數設定")

        body = {
            "schema": "crypto-strategy-github-state-v1",
            "saved_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "payload": _json_safe(payload),
        }

        raw = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        compressed = gzip.compress(raw, compresslevel=6)
        encoded = base64.b64encode(compressed).decode("ascii")

        # Retry once when SHA changed between GET and PUT.
        last_error = None
        for attempt in range(2):
            meta = self._get_file_meta(key)
            data = {
                "message": f"Update {self._object_path(key)}",
                "content": encoded,
                "branch": self.branch,
            }
            if meta and meta.get("sha"):
                data["sha"] = meta["sha"]

            response = self.session.put(
                self._content_url(key),
                json=data,
                timeout=self.timeout,
            )

            if response.status_code in (200, 201):
                result = response.json()
                content_info = result.get("content", {}) or {}
                return {
                    "ok": True,
                    "path": self._object_path(key),
                    "sha": content_info.get("sha"),
                    "bytes": len(compressed),
                    "saved_at_utc": body["saved_at_utc"],
                }

            last_error = RuntimeError(
                f"GitHub PUT 失敗 HTTP {response.status_code}: "
                f"{self._error_text(response)}"
            )

            if response.status_code in (409, 422) and attempt == 0:
                time.sleep(0.4)
                continue

            break

        raise last_error or RuntimeError("GitHub 儲存失敗")

    def load(self, key: str) -> Optional[dict]:
        if not self.configured:
            raise RuntimeError("GitHub 狀態儲存尚未完成環境變數設定")

        meta = self._get_file_meta(key)
        if not meta:
            return None

        content = meta.get("content")
        encoding = meta.get("encoding")

        if not content or encoding != "base64":
            # Download URL fallback, for responses where inline content is omitted.
            download_url = meta.get("download_url")
            if not download_url:
                raise RuntimeError("GitHub 狀態檔沒有可讀取內容")

            response = self.session.get(download_url, timeout=self.timeout)
            if response.status_code != 200:
                raise RuntimeError(
                    f"GitHub download 失敗 HTTP {response.status_code}"
                )
            compressed = response.content
        else:
            compressed = base64.b64decode(content.replace("\n", ""))

        body = json.loads(gzip.decompress(compressed).decode("utf-8"))

        if not isinstance(body, dict):
            raise RuntimeError("GitHub 雲端狀態格式錯誤")

        return body

    def delete(self, key: str) -> bool:
        if not self.configured:
            raise RuntimeError("GitHub 狀態儲存尚未完成環境變數設定")

        meta = self._get_file_meta(key)
        if not meta:
            return False

        response = self.session.delete(
            self._content_url(key),
            json={
                "message": f"Delete {self._object_path(key)}",
                "sha": meta["sha"],
                "branch": self.branch,
            },
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return True

        raise RuntimeError(
            f"GitHub DELETE 失敗 HTTP {response.status_code}: "
            f"{self._error_text(response)}"
        )
