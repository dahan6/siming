#!/usr/bin/env python3
"""墨城防火墙: Prometheus metrics exporter

在 HTTP :9101/metrics 暴露指标，供 Prometheus 抓取。

指标:
  mocheng_flows_scored_total        - 累计打分流量数
  mocheng_alerts_total{prio}        - 累计告警数（按级别）
  mocheng_blocks_active             - 当前活跃阻断规则数
  mocheng_model_tau                 - 当前异常阈值 τ
  mocheng_model_fpr                 - 最近验证 FPR
  mocheng_ndr_targets_tracked       - NDR 追踪的目标数
  mocheng_scan_duration_seconds     - 上次扫描耗时
  mocheng_flow_score{quantile}      - 打分分布（p50/p95/p99）

用法:
  from fw_metrics import MetricsExporter
  metrics = MetricsExporter()
  metrics.start()  # 启动 HTTP server（后台线程）
  metrics.record_flow(score=3.2)
  metrics.record_alert("P0")
"""
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import Counter, defaultdict


class MetricsExporter:
    """Prometheus 格式 metrics exporter"""

    def __init__(self, port=9101):
        self.port = port
        self._server = None
        self._thread = None

        # 计数器
        self.flows_scored = 0
        self.alerts = Counter()
        self.blocks_active = 0
        self.model_tau = 0.0
        self.model_fpr = 0.0
        self.ndr_targets = 0
        self.scan_duration = 0.0
        self.scan_timestamp = 0

        # 打分分布（滑动窗口近似）
        self._score_buckets = defaultdict(int)  # bucket → count
        self._lock = threading.Lock()

    def record_flow(self, score=0.0):
        with self._lock:
            self.flows_scored += 1
            bucket = round(score, 1)
            self._score_buckets[bucket] += 1
            # 限制内存
            if len(self._score_buckets) > 500:
                # 合并旧 bucket
                keys = sorted(self._score_buckets.keys())
                for k in keys[:200]:
                    del self._score_buckets[k]

    def record_alert(self, prio):
        with self._lock:
            self.alerts[prio] += 1

    def update_model_info(self, tau=0.0, fpr=0.0):
        with self._lock:
            self.model_tau = tau
            self.model_fpr = fpr

    def update_blocks(self, count):
        with self._lock:
            self.blocks_active = count

    def update_ndr(self, targets):
        with self._lock:
            self.ndr_targets = targets

    def record_scan(self, duration, n_flows):
        with self._lock:
            self.scan_duration = duration
            self.scan_timestamp = time.time()

    def _render(self):
        """渲染 Prometheus 文本格式"""
        lines = []
        with self._lock:
            # 流量打分
            lines.append(f"# HELP mocheng_flows_scored_total Total flows scored")
            lines.append(f"# TYPE mocheng_flows_scored_total counter")
            lines.append(f"mocheng_flows_scored_total {self.flows_scored}")

            # 告警（按级别）
            lines.append(f"# HELP mocheng_alerts_total Total alerts by priority")
            lines.append(f"# TYPE mocheng_alerts_total counter")
            for prio, count in sorted(self.alerts.items()):
                lines.append(f'mocheng_alerts_total{{prio="{prio}"}} {count}')

            # 阻断规则
            lines.append(f"# HELP mocheng_blocks_active Active block rules")
            lines.append(f"# TYPE mocheng_blocks_active gauge")
            lines.append(f"mocheng_blocks_active {self.blocks_active}")

            # 模型信息
            lines.append(f"# HELP mocheng_model_tau Anomaly threshold")
            lines.append(f"# TYPE mocheng_model_tau gauge")
            lines.append(f"mocheng_model_tau {self.model_tau}")

            lines.append(f"# HELP mocheng_model_fpr False positive rate")
            lines.append(f"# TYPE mocheng_model_fpr gauge")
            lines.append(f"mocheng_model_fpr {self.model_fpr}")

            # NDR
            lines.append(f"# HELP mocheng_ndr_targets_tracked NDR tracked targets")
            lines.append(f"# TYPE mocheng_ndr_targets_tracked gauge")
            lines.append(f"mocheng_ndr_targets_tracked {self.ndr_targets}")

            # 扫描耗时
            lines.append(f"# HELP mocheng_scan_duration_seconds Last scan duration")
            lines.append(f"# TYPE mocheng_scan_duration_seconds gauge")
            lines.append(f"mocheng_scan_duration_seconds {self.scan_duration}")

            # 打分分布近似
            scores = []
            for bucket, count in self._score_buckets.items():
                scores.extend([bucket] * min(count, 100))
            if scores:
                scores.sort()
                n = len(scores)
                for q, label in [(0.5, "p50"), (0.95, "p95"), (0.99, "p99")]:
                    val = scores[int(n * q)]
                    lines.append(f'mocheng_flow_score{{quantile="{label}"}} {val}')

        return "\n".join(lines) + "\n"

    def start(self):
        """启动 HTTP server（后台线程）"""
        renderer = self._render

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = renderer().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"MoCheng Firewall Metrics\nSee /metrics\n")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # 静默

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"[metrics] Prometheus exporter on :{self.port}/metrics")

    def stop(self):
        if self._server:
            self._server.shutdown()
