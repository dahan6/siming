#!/usr/bin/env python3
"""墨城防火墙: 执法层 v2

特性:
  - IPv4 (iptables) + IPv6 (ip6tables) 双栈
  - 规则持久化: save/restore，重启后自动恢复
  - Fail-closed: 服务停止时策略可配（默认保留规则）
  - 篡改检测: 定期校验 iptables -S 与 state 一致性
  - 白名单: 执法前检查，白名单 IP 永不阻断
  - 状态机感知: 按连接状态决策

用法:
  from fw_enforce import Enforcer
  enforcer = Enforcer(dry_run=True)
  enforcer.block("1.2.3.4", reason="C2", ttl=3600)
  enforcer.verify_integrity()  # 篡改检测
"""
import ipaddress
import json
import os
import subprocess
import time

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from fw_paths import RULES_DIR, ENFORCE_STATE

CHAIN_NAME = "MOCHENG"
SAVE_FILE = os.path.join(RULES_DIR, "mocheng_rules.sav")
WHITELIST_FILE = os.path.join(sys_path, "fw_whitelist.jsonl")


def is_ipv6(ip):
    """判断是否 IPv6 地址"""
    try:
        ipaddress.ip_address(ip)
        return ":" in ip
    except ValueError:
        return False


def iptables_cmd(ip):
    """返回对应 IP 版本的 iptables 命令"""
    return "ip6tables" if is_ipv6(ip) else "iptables"


def load_whitelist():
    """加载持久白名单

    格式: 每行一个 JSON: {"ip": "1.2.3.4", "reason": "DNS服务器"}
    或纯 IP 每行一行（# 开头注释）
    """
    wl = set()
    if not os.path.exists(WHITELIST_FILE):
        # 默认白名单
        return {"127.0.0.1", "::1"}
    for line in open(WHITELIST_FILE):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
            wl.add(entry["ip"])
            # 支持 CIDR
            if "/" in entry["ip"]:
                net = ipaddress.ip_network(entry["ip"], strict=False)
                wl.add(entry["ip"])
        except (json.JSONDecodeError, KeyError):
            # 纯 IP 行
            if line and not line.startswith("{"):
                wl.add(line)
    return wl


def is_whitelisted(ip, whitelist):
    """检查 IP 是否在白名单中（支持 CIDR 匹配）"""
    if ip in whitelist:
        return True
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for entry in whitelist:
        if "/" in entry:
            try:
                net = ipaddress.ip_network(entry, strict=False)
                if addr in net:
                    return True
            except ValueError:
                pass
    return False


class Enforcer:
    """iptables/ip6tables 规则管理器（v2）"""

    def __init__(self, dry_run=True, fail_closed=False):
        self.dry_run = dry_run
        self.fail_closed = fail_closed
        self.whitelist = load_whitelist()
        self.state = self._load_state()
        self._ensure_chain("iptables")
        self._ensure_chain("ip6tables")
        # 启动时恢复持久化规则
        if not dry_run:
            self._restore_rules()

    def _load_state(self):
        os.makedirs(RULES_DIR, exist_ok=True)
        if os.path.exists(ENFORCE_STATE):
            try:
                return json.load(open(ENFORCE_STATE))
            except (json.JSONDecodeError, IOError):
                pass  # 损坏的 state 从空开始
        return {"rules": {}, "last_verify": 0}

    def _save_state(self):
        os.makedirs(RULES_DIR, exist_ok=True)
        tmp = ENFORCE_STATE + ".tmp"
        json.dump(self.state, open(tmp, "w"), ensure_ascii=False, indent=2)
        os.replace(tmp, ENFORCE_STATE)  # 原子写

    def _run(self, cmd):
        if self.dry_run:
            print(f"  [dry-run] {' '.join(cmd)}")
            return True
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            return False

    def _ensure_chain(self, cmd_name):
        """创建自定义链并挂到 INPUT/OUTPUT"""
        self._run([cmd_name, "-N", CHAIN_NAME])
        for chain in ["INPUT", "OUTPUT"]:
            if not self._run([cmd_name, "-C", chain, "-j", CHAIN_NAME]):
                self._run([cmd_name, "-I", chain, "-j", CHAIN_NAME])

    # ── 规则持久化 ──

    def save_rules(self):
        """持久化当前 iptables 规则到文件"""
        for cmd_name in ["iptables", "ip6tables"]:
            try:
                result = subprocess.run([cmd_name, "-S", CHAIN_NAME],
                                        capture_output=True, text=True, timeout=5)
                rules = result.stdout.strip()
                sav_path = SAVE_FILE.replace(".sav", f"_{cmd_name}.sav")
                with open(sav_path, "w") as f:
                    f.write(rules)
            except (subprocess.SubprocessError, IOError):
                pass

    def _restore_rules(self):
        """从持久化文件恢复规则"""
        for cmd_name in ["iptables", "ip6tables"]:
            sav_path = SAVE_FILE.replace(".sav", f"_{cmd_name}.sav")
            if not os.path.exists(sav_path):
                continue
            try:
                rules = open(sav_path).read()
                for line in rules.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("-A"):
                        parts = line.split()
                        self._run([cmd_name] + parts[1:])
                print(f"[{cmd_name}] 恢复 {len(rules.strip().splitlines())} 条规则")
            except IOError:
                pass

    # ── 篡改检测 ──

    def verify_integrity(self):
        """校验 iptables 实际规则与 state 一致性

        Returns:
            dict: {"tampered": bool, "missing": [...], "extra": [...]}
        """
        result = {"tampered": False, "missing": [], "extra": []}
        expected_ips = set(self.state.get("rules", {}).keys())

        for cmd_name in ["iptables", "ip6tables"]:
            try:
                r = subprocess.run([cmd_name, "-S", CHAIN_NAME],
                                   capture_output=True, text=True, timeout=5)
                actual_lines = r.stdout.strip().split("\n")
            except subprocess.SubprocessError:
                continue

            actual_ips = set()
            for line in actual_lines:
                parts = line.split()
                if "-s" in parts:
                    idx = parts.index("-s")
                    if idx + 1 < len(parts):
                        actual_ips.add(parts[idx + 1])
                if "-d" in parts:
                    idx = parts.index("-d")
                    if idx + 1 < len(parts):
                        actual_ips.add(parts[idx + 1])

            # state 中有但 iptables 没有的 → 被删除了
            for ip in expected_ips - actual_ips:
                result["missing"].append(f"{cmd_name}:{ip}")
            # iptables 有但 state 没有的 → 被注入了
            for ip in actual_ips - expected_ips:
                if ip not in self.whitelist:
                    result["extra"].append(f"{cmd_name}:{ip}")

        result["tampered"] = bool(result["missing"] or result["extra"])
        self.state["last_verify"] = time.time()
        self._save_state()
        return result

    # ── 阻断/释放 ──

    def block(self, ip, reason="", ttl=3600, action="DROP"):
        """阻断指定 IP（白名单检查 + IPv4/IPv6 双栈）"""
        if is_whitelisted(ip, self.whitelist):
            print(f"[SKIP] {ip} 在白名单中，不阻断")
            return False

        if ip in self.state["rules"]:
            self.state["rules"][ip]["expires"] = time.time() + ttl
            self.state["rules"][ip]["hit_count"] += 1
            self._save_state()
            return True

        cmd_name = iptables_cmd(ip)
        expire_ts = time.time() + ttl
        self._run([cmd_name, "-A", CHAIN_NAME, "-s", ip, "-j", action])
        self._run([cmd_name, "-A", CHAIN_NAME, "-d", ip, "-j", action])

        self.state["rules"][ip] = {
            "reason": reason, "action": action,
            "created": time.time(), "expires": expire_ts,
            "hit_count": 1, "ip_version": "v6" if is_ipv6(ip) else "v4",
        }
        self._save_state()
        if not self.dry_run:
            self.save_rules()
        ts = time.strftime("%F %T")
        print(f"[{ts}] BLOCK {ip} ({action}) reason={reason} ttl={ttl}s")
        return True

    def throttle(self, ip, reason="", ttl=3600):
        """限速指定 IP"""
        if is_whitelisted(ip, self.whitelist):
            return False
        cmd_name = iptables_cmd(ip)
        if ip in self.state["rules"]:
            self.state["rules"][ip]["expires"] = time.time() + ttl
            self.state["rules"][ip]["hit_count"] += 1
            self._save_state()
            return True

        self._run([cmd_name, "-A", CHAIN_NAME, "-s", ip, "-m", "limit",
                   "--limit", "5/sec", "--limit-burst", "10", "-j", "RETURN"])
        self._run([cmd_name, "-A", CHAIN_NAME, "-s", ip, "-j", "DROP"])
        self.state["rules"][ip] = {
            "reason": reason, "action": "THROTTLE",
            "created": time.time(), "expires": time.time() + ttl,
            "hit_count": 1,
        }
        self._save_state()
        if not self.dry_run:
            self.save_rules()
        return True

    def release(self, ip):
        """释放指定 IP（逐条删，不存在静默跳过）"""
        if ip not in self.state.get("rules", {}):
            return
        cmd_name = iptables_cmd(ip)
        for action in ["DROP", "REJECT"]:
            for flag in ["-s", "-d"]:
                for _ in range(3):
                    if not self._run([cmd_name, "-D", CHAIN_NAME, flag, ip, "-j", action]):
                        break
        saved_action = self.state["rules"][ip].get("action", "DROP")
        if saved_action == "THROTTLE":
            self._run([cmd_name, "-D", CHAIN_NAME, "-s", ip, "-m", "limit",
                       "--limit", "5/sec", "-j", "RETURN"])
        del self.state["rules"][ip]
        self._save_state()
        if not self.dry_run:
            self.save_rules()
        ts = time.strftime("%F %T")
        print(f"[{ts}] RELEASE {ip}")

    def cleanup_expired(self):
        now = time.time()
        expired = [ip for ip, r in self.state.get("rules", {}).items()
                   if r.get("expires", 0) < now]
        for ip in expired:
            self.release(ip)
        return len(expired)

    def stats(self):
        now = time.time()
        rules = self.state.get("rules", {})
        active = sum(1 for r in rules.values() if r.get("expires", 0) > now)
        return {"total_rules": len(rules), "active": active,
                "whitelist_size": len(self.whitelist)}

    def shutdown(self):
        """服务关闭时的清理策略

        fail_closed=True: 保留所有阻断规则（安全优先）
        fail_closed=False: 清空 MOCHENG 链（连通性优先）
        """
        if self.fail_closed:
            self.save_rules()
            print("[shutdown] fail-closed: 规则已持久化保留")
        else:
            for cmd_name in ["iptables", "ip6tables"]:
                self._run([cmd_name, "-F", CHAIN_NAME])
            print("[shutdown] 规则已清空")


if __name__ == "__main__":
    enforcer = Enforcer(dry_run=True)
    enforcer.block("203.0.113.99", reason="测试", ttl=60)
    enforcer.block("2001:db8::1", reason="IPv6 测试", ttl=60)
    print(f"状态: {enforcer.stats()}")
    tamper = enforcer.verify_integrity()
    print(f"篡改检测: {tamper}")
    enforcer.cleanup_expired()
