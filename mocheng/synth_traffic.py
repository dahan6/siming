#!/usr/bin/env python3
"""墨城防火墙: 合成流量生成器

生成逼真的良性 + 攻击网络流事件，用于训练和验证检测管线。

良性模式（模拟一台正常工作站的 2 小时活动）:
  - DNS 查询（UDP 出站 → LAN DNS，频繁）
  - HTTPS 浏览（TCP 出站 → EXT，中等频率）
  - HTTP 浏览（TCP 出站 → EXT，偶尔）
  - NTP 同步（UDP 出站 → EXT，~20 分钟周期）
  - SSH 管理（TCP 出站 → LAN，罕见）
  - SMTP 邮件（TCP 出站 → EXT，罕见）
  - 包更新（TCP 出站 → EXT，罕见但大流量）

攻击模式:
  - A-portscan: 端口扫描（连续不同端口，DT0 爆发）
  - B-c2beacon: C2 信标（周期高端口外联）
  - C-exfil: 数据外泄（XL 大流量 → EXT 高端口）
  - D-lateral: 横向移动（内部 SMB/445 扫描）
  - E-dnstunnel: DNS 隧道（异常大量 DNS 查询）

用法: synth_traffic.py [总良性事件数=8000] [攻击注入=1]
输出: data/benign_flows.jsonl + data/attack_flows.jsonl
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import BASE_DIR, DATA_DIR

random.seed(42)

# ── IP 池 ──
LAN_IPS = [f"198.51.100.{i}" for i in range(2, 50)]
DNS_SERVER = "198.51.100.1"
EXT_IPS = [f"203.0.113.{i}" for i in range(1, 200)] + \
          [f"198.51.100.{i}" for i in range(1, 100)] + \
          [f"23.{i}.{i*3%255}.{i*7%255}" for i in range(10, 80)]
VPN_IPS = [f"192.0.2.{i}" for i in range(2, 20)]

# 本机 IP
LOCAL_IP = "198.51.100.100"


def ts_str(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch))


def make_flow(ts_epoch, proto, direction, src_ip, dst_ip,
              dst_port, src_port, n_bytes, action="accept"):
    return {
        "ts": ts_str(ts_epoch),
        "ts_epoch": round(ts_epoch, 3),
        "ts_sort": round(ts_epoch, 3),
        "proto": proto,
        "dir": direction,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "bytes": n_bytes,
        "action": action,
        "host": "fw",
    }


def gen_benign(n_events):
    """生成 n_events 条良性流量"""
    flows = []
    t = time.time() - 7200  # 从 2 小时前开始

    while len(flows) < n_events:
        roll = random.random()

        if roll < 0.30:
            # DNS 查询
            dst = random.choice(EXT_IPS)
            flows.append(make_flow(t, "udp", "out", LOCAL_IP, DNS_SERVER,
                                   53, random.randint(40000, 60000),
                                   random.randint(80, 200)))
            # 紧跟一个 HTTPS/HTTP 到解析到的 IP
            t2 = t + random.uniform(0.01, 0.5)
            if random.random() < 0.7:
                flows.append(make_flow(t2, "tcp", "out", LOCAL_IP, dst,
                                       443, random.randint(40000, 60000),
                                       random.randint(2000, 50000)))
            else:
                flows.append(make_flow(t2, "tcp", "out", LOCAL_IP, dst,
                                       80, random.randint(40000, 60000),
                                       random.randint(1000, 20000)))
            t += random.uniform(2, 30)

        elif roll < 0.45:
            # 独立 HTTPS 浏览
            dst = random.choice(EXT_IPS)
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                                   443, random.randint(40000, 60000),
                                   random.randint(3000, 80000)))
            t += random.uniform(5, 60)

        elif roll < 0.52:
            # HTTP 浏览
            dst = random.choice(EXT_IPS)
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                                   80, random.randint(40000, 60000),
                                   random.randint(1000, 20000)))
            t += random.uniform(5, 45)

        elif roll < 0.60:
            # NTP 同步（周期性）
            ntp_srv = random.choice(["129.6.15.30", "132.163.97.1"])
            flows.append(make_flow(t, "udp", "out", LOCAL_IP, ntp_srv,
                                   123, random.randint(40000, 50000),
                                   random.randint(60, 100)))
            t += random.uniform(1000, 1500)  # ~20 分钟

        elif roll < 0.68:
            # SSH 管理（内部）
            dst = random.choice(LAN_IPS)
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                                   22, random.randint(40000, 60000),
                                   random.randint(500, 5000)))
            t += random.uniform(30, 300)

        elif roll < 0.74:
            # 入站 HTTPS（服务器模式）
            src = random.choice(EXT_IPS)
            flows.append(make_flow(t, "tcp", "in", src, LOCAL_IP,
                                   443, random.randint(40000, 60000),
                                   random.randint(500, 3000)))
            t += random.uniform(10, 120)

        elif roll < 0.80:
            # SMTP 邮件
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, "203.0.113.50",
                                   587, random.randint(40000, 60000),
                                   random.randint(1000, 10000)))
            t += random.uniform(60, 600)

        elif roll < 0.86:
            # 包更新（大流量）
            dst = random.choice(EXT_IPS)
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                                   443, random.randint(40000, 60000),
                                   random.randint(50000, 500000)))
            t += random.uniform(300, 1800)

        elif roll < 0.92:
            # IMAP 收邮件
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, "203.0.113.60",
                                   993, random.randint(40000, 60000),
                                   random.randint(2000, 30000)))
            t += random.uniform(60, 300)

        else:
            # VPN 连接
            dst = random.choice(VPN_IPS)
            flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                                   443, random.randint(40000, 60000),
                                   random.randint(1000, 10000)))
            t += random.uniform(120, 600)

    return flows[:n_events]


def gen_attack_portscan(t_start):
    """A-portscan: 快速端口扫描"""
    flows = []
    target = random.choice(EXT_IPS)
    ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445,
             993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 9200]
    t = t_start
    for p in ports:
        flows.append(make_flow(t, "tcp", "out", LOCAL_IP, target,
                               p, random.randint(45000, 55000),
                               random.randint(40, 80)))
        t += random.uniform(0.005, 0.05)  # 极快爆发
    return flows, t


def gen_attack_c2beacon(t_start):
    """B-c2beacon: C2 信标——周期性高端口外联"""
    flows = []
    c2_srv = random.choice(EXT_IPS[:20])
    c2_port = random.choice([4444, 5555, 8443, 9999, 12345])
    t = t_start
    for _ in range(15):
        flows.append(make_flow(t, "tcp", "out", LOCAL_IP, c2_srv,
                               c2_port, random.randint(45000, 55000),
                               random.randint(100, 500)))
        t += random.uniform(28, 32)  # 精确 ~30 秒周期
    return flows, t


def gen_attack_exfil(t_start):
    """C-exfil: 数据外泄——大流量外联"""
    flows = []
    dst = random.choice(EXT_IPS)
    t = t_start
    for _ in range(5):
        flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                               random.choice([4444, 8080, 9999]),
                               random.randint(45000, 55000),
                               random.randint(500000, 5000000)))  # 0.5-5 MB
        t += random.uniform(0.5, 2)
    return flows, t


def gen_attack_lateral(t_start):
    """D-lateral: 横向移动——内部 SMB 扫描"""
    flows = []
    t = t_start
    for _ in range(10):
        dst = random.choice(LAN_IPS)
        flows.append(make_flow(t, "tcp", "out", LOCAL_IP, dst,
                               445, random.randint(45000, 55000),
                               random.randint(100, 500)))
        t += random.uniform(0.1, 0.5)
    # 尝试 RDP
    flows.append(make_flow(t, "tcp", "out", LOCAL_IP, random.choice(LAN_IPS),
                           3389, random.randint(45000, 55000),
                           random.randint(200, 600)))
    return flows, t


def gen_attack_dnstunnel(t_start):
    """E-dnstunnel: DNS 隧道——异常大量 DNS 查询"""
    flows = []
    t = t_start
    for _ in range(30):
        flows.append(make_flow(t, "udp", "out", LOCAL_IP, DNS_SERVER,
                               53, random.randint(40000, 60000),
                               random.randint(300, 600)))  # 异常大的 DNS 包
        t += random.uniform(0.02, 0.1)
    return flows, t


ATTACKS = {
    "portscan": gen_attack_portscan,
    "c2beacon": gen_attack_c2beacon,
    "exfil": gen_attack_exfil,
    "lateral": gen_attack_lateral,
    "dnstunnel": gen_attack_dnstunnel,
}


def main():
    n_benign = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    inject = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    # 良性流量
    benign = gen_benign(n_benign)
    benign_path = os.path.join(data_dir, "benign_flows.jsonl")
    with open(benign_path, "w") as f:
        for fl in benign:
            f.write(json.dumps(fl, ensure_ascii=False) + "\n")
    print(f"良性流量 {len(benign)} 条 -> {benign_path}")

    # 攻击流量（每类各注入 inject 轮，从良性流量中段时间点开始）
    if inject:
        mid_t = benign[len(benign)//2]["ts_epoch"] + 60
        attack_path = os.path.join(data_dir, "attack_flows.jsonl")
        all_attacks = []
        t = mid_t
        with open(attack_path, "w") as f:
            for name, gen in ATTACKS.items():
                for _ in range(inject):
                    flows, t_end = gen(t)
                    for fl in flows:
                        fl["attack_type"] = name
                        f.write(json.dumps(fl, ensure_ascii=False) + "\n")
                        all_attacks.append((name, fl))
                    t = t_end + random.uniform(120, 300)  # 攻击间隔
        print(f"攻击流量 {len(all_attacks)} 条（{len(ATTACKS)} 类 × {inject} 轮）-> {attack_path}")

        # 混合数据（训练用良性，验证用攻击）
        mixed_path = os.path.join(data_dir, "mixed_flows.jsonl")
        combined = benign + [a[1] for a in all_attacks]
        combined.sort(key=lambda f: f["ts_epoch"])
        with open(mixed_path, "w") as f:
            for fl in combined:
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")
        print(f"混合流量 {len(combined)} 条 -> {mixed_path}")


if __name__ == "__main__":
    main()
