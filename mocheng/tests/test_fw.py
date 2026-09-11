#!/usr/bin/env python3
"""墨城防火墙单元测试

覆盖: fw_tokens / fw_patterns / fw_enforce / fw_calibrate / fw_alert
运行: python -m pytest tests/ -v  或  python tests/test_fw.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fw_tokens import (flow_to_tokens, net_class, port_class, size_class,
                       dt_bucket, parse_flow_line)
from fw_patterns import FwPatternDB, _tokmap
from fw_enforce import Enforcer
from fw_calibrate import slot_of
from fw_alert import AlertSink


class TestFwTokens(unittest.TestCase):
    """fw_tokens: 网络流 → 7-token 离散化"""

    def test_net_class(self):
        self.assertEqual(net_class("127.0.0.1"), "LOOPBACK")
        self.assertEqual(net_class("198.51.100.100"), "LAN")
        self.assertEqual(net_class("10.0.0.1"), "LAN")
        self.assertEqual(net_class("192.0.2.5"), "VPN")
        self.assertEqual(net_class("8.8.8.8"), "EXT")
        self.assertEqual(net_class("?"), "UNKNOWN")

    def test_port_class(self):
        self.assertEqual(port_class(443), "HTTPS")
        self.assertEqual(port_class(22), "SSH")
        self.assertEqual(port_class(53), "DNS")
        self.assertEqual(port_class(445), "SMB")
        self.assertEqual(port_class(80), "HTTP")
        self.assertEqual(port_class(999), "WELLKNOWN")
        self.assertEqual(port_class(4444), "HIGHPORT")
        self.assertEqual(port_class(50000), "EPHEMERAL")
        self.assertEqual(port_class(0), "NONE")

    def test_size_class(self):
        self.assertEqual(size_class(0), "ZERO")
        self.assertEqual(size_class(100), "S")
        self.assertEqual(size_class(1000), "M")
        self.assertEqual(size_class(50000), "L")
        self.assertEqual(size_class(500000), "XL")

    def test_dt_bucket(self):
        self.assertEqual(dt_bucket(0), "DT0")
        self.assertEqual(dt_bucket(5), "DT1")
        self.assertEqual(dt_bucket(50), "DT2")
        self.assertEqual(dt_bucket(500), "DT3")
        self.assertEqual(dt_bucket(5000), "DT4")
        self.assertEqual(dt_bucket(30000), "DT5")
        self.assertEqual(dt_bucket(120000), "DT6")

    def test_flow_to_tokens(self):
        flow = {"proto": "tcp", "dir": "out", "src_ip": "198.51.100.100",
                "dst_ip": "8.8.8.8", "dst_port": 443, "bytes": 4999}
        tokens = flow_to_tokens(flow, 100)
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[0], "PROTO:TCP")
        self.assertEqual(tokens[1], "DIR:OUT")
        self.assertEqual(tokens[2], "SRCNET:LAN")
        self.assertEqual(tokens[3], "DSTNET:EXT")
        self.assertEqual(tokens[4], "PORTCLS:HTTPS")
        self.assertEqual(tokens[5], "SIZECLS:M")
        self.assertEqual(tokens[6], "DT3")

    def test_parse_flow_line(self):
        line = '{"proto":"udp","dir":"out","src_ip":"10.0.0.1","dst_ip":"8.8.8.8","dst_port":53,"bytes":100}'
        flow = parse_flow_line(line)
        self.assertIsNotNone(flow)
        self.assertEqual(flow["proto"], "udp")
        self.assertIsNone(parse_flow_line("not json"))
        self.assertIsNone(parse_flow_line(""))


class TestFwPatterns(unittest.TestCase):
    """fw_patterns: 模式匹配引擎"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        self.tmp.write(json.dumps({
            "id": "TEST-001", "type": "pattern", "technique": "T1046",
            "name": "测试模式", "severity": 3, "action": "DROP",
            "match": {"proto": "TCP", "dir": "OUT", "dstnet": "EXT",
                      "portcls_in": ["HIGHPORT"]},
            "precision": "category", "context_required": False,
        }) + "\n")
        self.tmp.close()
        self.db = FwPatternDB(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_match_hit(self):
        tokens = ["PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
                  "PORTCLS:HIGHPORT", "SIZECLS:S", "DT0"]
        hits = self.db.match(tokens)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "TEST-001")

    def test_match_miss(self):
        tokens = ["PROTO:UDP", "DIR:OUT", "SRCNET:LAN", "DSTNET:LAN",
                  "PORTCLS:DNS", "SIZECLS:S", "DT3"]
        hits = self.db.match(tokens)
        self.assertEqual(len(hits), 0)

    def test_tokmap(self):
        tokens = ["PROTO:TCP", "DIR:OUT", "DT3"]
        tm = _tokmap(tokens)
        self.assertEqual(tm["PROTO"], "TCP")
        self.assertEqual(tm["DIR"], "OUT")
        self.assertEqual(tm["DT"], "DT3")

    def test_context_required(self):
        # 添加一个 context_required=True 的模式
        self.db.add({
            "id": "TEST-002", "type": "pattern", "technique": "T1071",
            "name": "需上下文", "severity": 5, "action": "DROP",
            "match": {"proto": "TCP", "dir": "OUT", "dstnet": "EXT",
                      "portcls_in": ["HIGHPORT"], "dt_in": ["DT3"]},
            "context_required": True,
        })
        # 无 DT3 → 不命中
        tokens_no_dt = ["PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
                        "PORTCLS:HIGHPORT", "SIZECLS:S", "DT0"]
        hits = [h for h in self.db.match(tokens_no_dt) if h["id"] == "TEST-002"]
        self.assertEqual(len(hits), 0)
        # 有 DT3 → 命中
        tokens_dt3 = ["PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
                      "PORTCLS:HIGHPORT", "SIZECLS:S", "DT3"]
        hits = [h for h in self.db.match(tokens_dt3) if h["id"] == "TEST-002"]
        self.assertEqual(len(hits), 1)


class TestFwEnforce(unittest.TestCase):
    """fw_enforce: iptables 执法层"""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        import fw_enforce
        self._orig_rules_dir = fw_enforce.RULES_DIR
        self._orig_enforce_state = fw_enforce.ENFORCE_STATE
        fw_enforce.RULES_DIR = self._tmpdir
        fw_enforce.ENFORCE_STATE = os.path.join(self._tmpdir, "state.json")

    def tearDown(self):
        import fw_enforce, shutil
        fw_enforce.RULES_DIR = self._orig_rules_dir
        fw_enforce.ENFORCE_STATE = self._orig_enforce_state
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_dry_run_block(self):
        enforcer = Enforcer(dry_run=True)
        enforcer.block("203.0.113.99", reason="测试", ttl=60)
        self.assertIn("203.0.113.99", enforcer.state["rules"])
        self.assertEqual(enforcer.state["rules"]["203.0.113.99"]["action"], "DROP")

    def test_release(self):
        enforcer = Enforcer(dry_run=True)
        enforcer.block("198.51.100.5", reason="测试", ttl=60)
        enforcer.release("198.51.100.5")
        self.assertNotIn("198.51.100.5", enforcer.state["rules"])

    def test_ttl_expiry(self):
        enforcer = Enforcer(dry_run=True)
        enforcer.block("203.0.113.1", reason="测试", ttl=0)
        import time; time.sleep(0.1)
        n = enforcer.cleanup_expired()
        self.assertEqual(n, 1)

    def test_stats(self):
        enforcer = Enforcer(dry_run=True)
        enforcer.block("1.2.3.4", reason="测试", ttl=3600)
        stats = enforcer.stats()
        self.assertEqual(stats["total_rules"], 1)
        self.assertEqual(stats["active"], 1)


class TestSlotOf(unittest.TestCase):
    """fw_calibrate: slot_of"""

    def test_slot_of(self):
        self.assertEqual(slot_of("PROTO:TCP"), "PROTO")
        self.assertEqual(slot_of("DSTNET:EXT"), "DSTNET")
        self.assertEqual(slot_of("DT3"), "DT")
        self.assertEqual(slot_of("PORTCLS:HTTPS"), "PORTCLS")


class TestAlertSink(unittest.TestCase):
    """fw_alert: 告警外发"""

    def test_file_sink(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            config = {"daemon": {"alerts": path}, "alert": {"method": "file"}}
            sink = AlertSink(config)
            sink.emit({"prio": "P0", "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8",
                       "dst_port": 443, "s_ev": 7.5, "ewma": 3.2})
            sink.close()
            lines = open(path).readlines()
            self.assertEqual(len(lines), 1)
            alert = json.loads(lines[0])
            self.assertEqual(alert["prio"], "P0")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
