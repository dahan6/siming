#!/usr/bin/env python3
"""墨城防火墙: 告警外发模块

支持三种外发方式（可组合）:
  - file:    写入 JSONL 文件（默认）
  - syslog:  发送到 syslog（facility/tag 可配）
  - webhook: POST JSON 到指定 URL

用法:
  from fw_alert import AlertSink
  sink = AlertSink(config)
  sink.emit(alert_dict)
  sink.close()
"""
import json
import logging
import logging.handlers
import os
import sys
import urllib.request
import urllib.error

from fw_paths import ALERTS_FILE


class AlertSink:
    """告警外发：file + syslog + webhook"""

    def __init__(self, config=None):
        config = config or {}
        alert_cfg = config.get("alert", {})
        self.method = alert_cfg.get("method", "file")
        self._sinks = []

        # file sink
        if self.method in ("file", "all"):
            path = config.get("daemon", {}).get("alerts", ALERTS_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._file = open(path, "a")
            self._sinks.append("file")
        else:
            self._file = None

        # syslog sink（带优雅降级）
        self._syslog = None
        if self.method in ("syslog", "all"):
            facility_name = alert_cfg.get("syslog_facility", "local0")
            tag = alert_cfg.get("syslog_tag", "mocheng")
            facility = getattr(logging.handlers.SysLogHandler,
                               f"LOG_{facility_name.upper()}",
                               logging.handlers.SysLogHandler.LOG_LOCAL0)
            # 尝试 /dev/log（Linux）→ /var/run/syslog（macOS）
            syslog_addr = None
            for addr in ["/dev/log", "/var/run/syslog", "/var/run/systemd/journal/dev-log"]:
                if os.path.exists(addr):
                    syslog_addr = addr
                    break
            if syslog_addr:
                try:
                    self._syslog = logging.getLogger("mocheng")
                    self._syslog.setLevel(logging.INFO)
                    handler = logging.handlers.SysLogHandler(
                        address=syslog_addr, facility=facility)
                    handler.setFormatter(logging.Formatter(f"{tag}: %(message)s"))
                    self._syslog.addHandler(handler)
                    self._sinks.append("syslog")
                except (OSError, FileNotFoundError):
                    self._syslog = None
                    print("[WARN] syslog 初始化失败，退化为 file-only")
            else:
                print("[WARN] syslog socket 不存在，退化为 file-only")

        # webhook sink
        if self.method in ("webhook", "all"):
            self._webhook_url = alert_cfg.get("webhook_url", "")
            self._webhook_timeout = alert_cfg.get("webhook_timeout", 5)
            if self._webhook_url:
                self._sinks.append("webhook")
        else:
            self._webhook_url = ""

    def emit(self, alert):
        """发送一条告警到所有已启用的 sink"""
        # file
        if self._file:
            self._file.write(json.dumps(alert, ensure_ascii=False) + "\n")
            self._file.flush()

        # syslog
        if self._syslog:
            prio = alert.get("prio", "?")
            src = alert.get("src_ip", "?")
            dst = alert.get("dst_ip", "?")
            port = alert.get("dst_port", "?")
            reason = ""
            if alert.get("proto_hit"):
                reason = f"proto={alert['proto_hit']['technique']}"
            elif alert.get("patterns"):
                reason = alert["patterns"][0].get("name", "")
            elif alert.get("fired_dims"):
                reason = ",".join(alert["fired_dims"][:3])
            self._syslog.info(
                f"[{prio}] {src}→{dst}:{port} {reason} "
                f"s_ev={alert.get('s_ev',0):.2f} ewma={alert.get('ewma',0):.2f}")

        # webhook
        if self._webhook_url:
            self._post_webhook(alert)

    def _post_webhook(self, alert):
        """POST JSON 到 webhook URL（非阻塞，失败静默）"""
        try:
            data = json.dumps(alert, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST")
            urllib.request.urlopen(req, timeout=self._webhook_timeout)
        except (urllib.error.URLError, OSError, ValueError):
            pass  # 告警外发失败不影响主流程

    def close(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
