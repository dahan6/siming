#!/usr/bin/env python3
"""合成攻击序列测试：验证 adaptive_detector 对 rebirth/提权/窃取/C2 的检测

用隐翅虫 v4 文档中的行为序列构造合成事件，喂给 AdaptiveDetector，
验证各检测器是否正确触发。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_detector import AdaptiveDetector


def make_event(et, proc, argv, parent, uid, dst, pc, dt):
    """构造 8-token 事件"""
    return [
        f"ET:{et}", f"PROC:{proc}", f"ARGV:{argv}",
        f"PARENT:{parent}", f"UID:{uid}", f"DST:{dst}",
        f"PC:{pc}", f"DT{dt}",
    ]


def test_morph_transform():
    """测试优雅退休重生序列：cp→chmod→setsid→rm"""
    print("\n=== 测试: rebirth 优雅退休 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)
    # 前置正常事件
    for i in range(20):
        ev = make_event("EXEC", "sleep", "N1-", "bash", "1000", "NONE", "NONE", "1")
        det.update(ev, ts=f"2026-07-31T10:{i:02d}:00")

    # rebirth 序列
    events = [
        make_event("EXEC", "cp", "N2P", "bash", "1000", "NONE", "TMP", "1"),
        make_event("EXEC", "chmod", "N2P", "bash", "1000", "NONE", "TMP", "0"),
        make_event("EXEC", "setsid", "N1P", "bash", "1000", "NONE", "TMP", "1"),
        make_event("EXEC", "rm", "N1P", "bash", "1000", "NONE", "TMP", "4"),
    ]
    alerts = []
    for ev in events:
        alerts.extend(det.update(ev, ts=f"2026-07-31T11:00:0{len(alerts)}"))

    rebirth_hits = [a for a in alerts if a["type"] == "MORPH_TRANSFORM"]
    if rebirth_hits:
        print(f"  ✅ MORPH_TRANSFORM 触发 ({len(rebirth_hits)} 次)")
        print(f"     detail: {rebirth_hits[0]['detail']}")
    else:
        print(f"  ❌ MORPH_TRANSFORM 未触发")
    return len(rebirth_hits) > 0


def test_disguise_c2():
    """测试伪装进程 C2 外联"""
    print("\n=== 测试: 伪装 C2 外联 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)

    # node 外联 EXT:HIGH
    ev = make_event("CONN", "node", "ARGV0", "?", "1000", "EXT:HIGH", "NONE", "0")
    alerts = det.update(ev, ts="2026-07-31T12:00:00")

    c2_hits = [a for a in alerts if a["type"] == "DISGUISE_C2"]
    if c2_hits:
        print(f"  ✅ DISGUISE_C2 触发")
        print(f"     detail: {c2_hits[0]['detail']}")
    else:
        print(f"  ❌ DISGUISE_C2 未触发")

    # 测试全部伪装名池
    print("  测试全部伪装名池:")
    all_pass = True
    for name in ["node", "python3", "go-build", "cargo-watch", "dotnet",
                 "ruby", "java", "kubectl", "terraform"]:
        det2 = AdaptiveDetector(window_size=50, cooldown=1)
        ev = make_event("CONN", name, "ARGV0", "?", "1000", "EXT:HIGH", "NONE", "0")
        a = det2.update(ev, ts="2026-07-31T12:01:00")
        ok = any(x["type"] == "DISGUISE_C2" for x in a)
        print(f"    {'✅' if ok else '❌'} {name}")
        if not ok:
            all_pass = False
    return len(c2_hits) > 0 and all_pass


def test_suid_privesc():
    """测试 SUID python3 提权"""
    print("\n=== 测试: SUID python3 提权 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)

    # 正常 python3 执行（UID:1000，非提权）
    ev = make_event("EXEC", "python3", "N1P", "bash", "1000", "NONE", "NONE", "1")
    alerts = det.update(ev, ts="2026-07-31T13:00:00")
    privesc = [a for a in alerts if a["type"] == "SUID_PRIVESC"]
    if privesc:
        print(f"  ❌ 误报：UID:1000 的 python3 不应触发")
    else:
        print(f"  ✅ UID:1000 python3 正确不触发")

    # SUID python3 UID:0 无 sudo
    ev = make_event("EXEC", "python3", "N1P", "bash", "0", "NONE", "NONE", "0")
    alerts = det.update(ev, ts="2026-07-31T13:00:01")
    privesc = [a for a in alerts if a["type"] == "SUID_PRIVESC"]
    if privesc:
        print(f"  ✅ SUID_PRIVESC 触发")
        print(f"     detail: {privesc[0]['detail']}")
    else:
        print(f"  ❌ SUID_PRIVESC 未触发")

    # sudo python3 UID:0（正常提权，不应触发）
    det2 = AdaptiveDetector(window_size=200, cooldown=1)
    ev1 = make_event("EXEC", "sudo", "N1P", "bash", "0", "NONE", "NONE", "0")
    det2.update(ev1, ts="2026-07-31T13:01:00")
    ev2 = make_event("EXEC", "python3", "N1P", "bash", "0", "NONE", "NONE", "0")
    alerts = det2.update(ev2, ts="2026-07-31T13:01:01")
    privesc = [a for a in alerts if a["type"] == "SUID_PRIVESC"]
    if privesc:
        print(f"  ⚠️  sudo python3 触发了（可能误报，需看 context）")
    else:
        print(f"  ✅ sudo python3 正确不触发（有 sudo 在链上）")

    return len(privesc) == 0  # 最后一个应该是 False


def test_exfil_memory():
    """测试内存直传窃取序列：cat passwd → python3 POST"""
    print("\n=== 测试: 内存直传窃取 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)

    # 前置 sleep
    for i in range(5):
        det.update(make_event("EXEC", "sleep", "N1-", "bash", "1000", "NONE", "NONE", "1"),
                   ts=f"2026-07-31T14:00:0{i}")

    # cat /etc/passwd
    ev1 = make_event("EXEC", "cat", "N1P", "bash", "1000", "NONE", "ETC_PASSWD", "1")
    alerts1 = det.update(ev1, ts="2026-07-31T14:00:10")

    # python3 POST to EXT:HIGH
    ev2 = make_event("CONN", "python3", "ARGV0", "bash", "1000", "EXT:HIGH", "NONE", "1")
    alerts2 = det.update(ev2, ts="2026-07-31T14:00:11")

    all_alerts = alerts1 + alerts2
    c2_hits = [a for a in all_alerts if a["type"] == "DISGUISE_C2"]
    if c2_hits:
        print(f"  ✅ 窃取→C2 传输检测到（python3 外联 EXT:HIGH）")
        print(f"     detail: {c2_hits[0]['detail']}")
    else:
        print(f"  ❌ 未检测到窃取→C2")

    return len(c2_hits) > 0


def test_recon_uniform():
    """测试均匀侦察轮换"""
    print("\n=== 测试: 均匀侦察轮换 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)

    # 构造均匀轮换：ss→sleep→ip→sleep→ps→sleep→ls→sleep→find→sleep→free→sleep→df→sleep→uptime→sleep
    cmds = ["ss", "ip", "ps", "ls", "find", "free", "df", "uptime"]
    alerts_all = []
    for i in range(3):  # 循环 3 次
        for cmd in cmds:
            ev = make_event("EXEC", cmd, "N1-", "bash", "1000", "NONE", "NONE", "3")
            alerts_all.extend(det.update(ev, ts=f"2026-07-31T15:{i:02d}:{cmds.index(cmd)*5:02d}"))
            # 插 sleep
            ev_sleep = make_event("EXEC", "sleep", "N1-", "bash", "1000", "NONE", "NONE", "1")
            det.update(ev_sleep, ts=f"2026-07-31T15:{i:02d}:{cmds.index(cmd)*5+1:02d}")

    recon_hits = [a for a in alerts_all if a["type"] == "RECON_UNIFORM"]
    if recon_hits:
        print(f"  ✅ RECON_UNIFORM 触发 ({len(recon_hits)} 次)")
        print(f"     detail: {recon_hits[-1]['detail'][:80]}")
    else:
        print(f"  ❌ RECON_UNIFORM 未触发")

    return len(recon_hits) > 0


def test_sleep_stepping():
    """测试 sleep 步进模式"""
    print("\n=== 测试: sleep 步进交替 ===")
    det = AdaptiveDetector(window_size=200, cooldown=1)

    # 构造 sleep→动作→sleep→动作 模式
    actions = ["head", "cat", "ls", "ps", "ss", "free", "df", "uptime",
               "head", "cat", "ls", "ps"]
    alerts_all = []
    for i, act in enumerate(actions):
        # sleep
        det.update(make_event("EXEC", "sleep", "N1-", "bash", "1000", "NONE", "NONE", "1"),
                   ts=f"2026-07-31T16:{i:02d}:00")
        # action
        alerts_all.extend(
            det.update(make_event("EXEC", act, "N1-", "bash", "1000", "NONE", "NONE", "0"),
                       ts=f"2026-07-31T16:{i:02d}:01"))

    sleep_hits = [a for a in alerts_all if a["type"] == "SLEEP_STEPPING"]
    if sleep_hits:
        print(f"  ✅ SLEEP_STEPPING 触发 ({len(sleep_hits)} 次)")
        print(f"     detail: {sleep_hits[-1]['detail']}")
    else:
        print(f"  ❌ SLEEP_STEPPING 未触发")

    return len(sleep_hits) > 0


def test_benign_baseline():
    """测试正常活动不触发高误报"""
    print("\n=== 测试: 良性活动低误报 ===")
    det = AdaptiveDetector(window_size=200, cooldown=50)

    # 正常系统活动（cron 每 5 分钟一次，混合管理命令偏斜分布）
    import random
    random.seed(42)
    alerts_all = []
    for i in range(500):
        # 偏斜分布：ls >> ss >> 其他
        r = random.random()
        if r < 0.4:
            cmd = "ls"
        elif r < 0.6:
            cmd = "cat"
        elif r < 0.7:
            cmd = "grep"
        elif r < 0.8:
            cmd = "ps"
        elif r < 0.85:
            cmd = "ss"
        elif r < 0.9:
            cmd = "systemctl"
        else:
            cmd = random.choice(["find", "df", "free", "uptime", "journalctl"])
        ev = make_event("EXEC", cmd, "N1-", "bash", "1000", "NONE", "NONE", str(random.choice([0,1,2,3])))
        alerts_all.extend(det.update(ev, ts=f"2026-07-31T17:{i//60:02d}:{i%60:02d}"))

    # 不应该触发 RECON_UNIFORM（分布偏斜，不均匀）
    recon_hits = [a for a in alerts_all if a["type"] == "RECON_UNIFORM"]
    if len(recon_hits) == 0:
        print(f"  ✅ 偏斜分布不触发 RECON_UNIFORM")
    else:
        print(f"  ⚠️  偏斜分布触发了 {len(recon_hits)} 次 RECON_UNIFORM")

    # 不应该触发 SLEEP_STEPPING（无 sleep 交替）
    sleep_hits = [a for a in alerts_all if a["type"] == "SLEEP_STEPPING"]
    if len(sleep_hits) == 0:
        print(f"  ✅ 无 sleep 交替不触发 SLEEP_STEPPING")
    else:
        print(f"  ⚠️  触发了 {len(sleep_hits)} 次 SLEEP_STEPPING")

    total = len(alerts_all)
    print(f"  总告警: {total}/500 ({total/500*100:.1f}%)")

    return len(recon_hits) == 0 and len(sleep_hits) == 0


def main():
    print("=" * 60)
    print("隐翅虫 v4 合成攻击序列检测测试")
    print("=" * 60)

    results = {}
    results["rebirth"] = test_morph_transform()
    results["c2"] = test_disguise_c2()
    results["privesc"] = test_suid_privesc()
    results["exfil"] = test_exfil_memory()
    results["recon"] = test_recon_uniform()
    results["sleep"] = test_sleep_stepping()
    results["benign"] = test_benign_baseline()

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    pass_cnt = 0
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
        if ok:
            pass_cnt += 1
    print(f"\n通过: {pass_cnt}/{len(results)}")


if __name__ == "__main__":
    main()
