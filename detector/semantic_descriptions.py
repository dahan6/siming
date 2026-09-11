#!/usr/bin/env python3
"""syscall/行为语义文本描述库

每个 token 映射到一段自然语言描述，用于预训练语义 embedding。
编码器学习后，语义相近的 token 在向量空间靠近。

例如：
  PROC:cat  → "read and display file contents, often used for file inspection"
  PROC:head → "display first lines of a file, used for quick file preview"
  → 编码后 embedding 相近（都是文件读取）

  PROC:ss   → "show network socket statistics, network reconnaissance tool"
  PROC:netstat → "display network connections and routing, network reconnaissance"
  → 编码后 embedding 相近（都是网络侦察）
"""

# ═══ PROC 语义描述 ═══
PROC_DESCRIPTIONS = {
    # ── 文件读取类（语义相近）──
    "cat":     "read and display file contents, often used for file inspection and data collection",
    "head":    "display first lines of a file, used for quick file preview or partial read",
    "tail":    "display last lines of a file, used for log monitoring or data extraction",
    "less":    "interactive file viewer, paginated file reading",
    "more":    "paginated file viewer, basic file reading",
    "dd":      "copy and convert files, raw data read or write including block devices",

    # ── 文件搜索类 ──
    "find":    "search for files in directory hierarchy, often used for discovery or SUID binary search",
    "locate":  "find files by name using pre-built database, fast file discovery",
    "which":   "locate executable command path, command discovery",
    "whereis": "locate binary source and manual page, command discovery",

    # ── 网络侦察类（语义相近）──
    "ss":      "show network socket statistics, active connection reconnaissance",
    "netstat": "display network connections and routing tables, network reconnaissance",
    "lsof":    "list open files and network connections, resource discovery",
    "ip":      "show or manipulate network interfaces and routing, network configuration reconnaissance",
    "ifconfig":"display or configure network interfaces, network reconnaissance",
    "arp":     "display and modify IP-to-MAC address cache, network discovery",
    "nmap":    "network scanner for port service and OS detection, active network reconnaissance",

    # ── 进程侦察类 ──
    "ps":      "report process status, running process enumeration",
    "top":     "display sorted process information, resource monitoring and process reconnaissance",
    "htop":    "interactive process viewer, detailed process enumeration",
    "pgrep":   "find processes by name, targeted process search",
    "pidof":   "find process ID by name, process identification",

    # ── 持久化类（语义相近）──
    "crontab": "schedule recurring task execution, cron job persistence mechanism",
    "at":      "schedule one-time future task execution, delayed task persistence",
    "atrm":    "remove scheduled at tasks, persistence cleanup or modification",
    "atq":     "list scheduled at tasks, persistence enumeration",
    "batch":   "queue tasks for batch execution when system load permits, deferred persistence",

    # ── 系统服务操控 ──
    "systemctl": "control systemd services, service management and persistence via systemd units",
    "tee":     "write to file and stdout simultaneously, used for redirecting output to system files",

    # ── 权限操作类 ──
    "sudo":    "execute command as superuser, legitimate privilege escalation tool",
    "su":      "switch user account, privilege switching mechanism",
    "pkexec":  "policy-kit execution, authenticated privilege escalation",
    "chmod":   "change file permissions, permission modification including SUID setting",
    "chown":   "change file ownership, ownership modification",
    "setcap":  "set file capabilities, fine-grained privilege assignment",
    "getcap":  "query file capabilities, capability enumeration for privilege discovery",

    # ── 网络工具类 ──
    "curl":    "transfer data from URLs, HTTP client often used for data exfiltration or C2",
    "wget":    "download files from web, HTTP file retrieval often used for payload download",
    "nc":      "netcat arbitrary TCP UDP connection, network tool for tunnels and backdoors",
    "ncat":    "enhanced netcat with SSL and access control, network tunnel tool",
    "ssh":     "secure shell remote login, remote access and lateral movement tool",
    "scp":     "secure copy over SSH, remote file transfer and lateral movement",
    "sftp":    "secure file transfer protocol, remote file access",
    "rsync":   "remote file synchronization, efficient file transfer between hosts",

    # ── 编码/加密类 ──
    "base64":  "encode or decode base64 data, obfuscation tool for hiding payloads",
    "openssl": "SSL TLS toolkit and crypto library, encryption and certificate operations",
    "xxd":     "hexadecimal dump of binary files, binary data inspection",

    # ── 进程控制类 ──
    "setsid":  "run program in new session, daemon creation and background persistence",
    "nohup":   "run command immune to hangups, persistent background execution",
    "kill":    "terminate process by signal, process termination",
    "pkill":   "kill processes by name, batch process termination",
    "systemd-executor": "systemd service executor, system service process spawning",

    # ── 文件操作类 ──
    "cp":      "copy files and directories, file duplication including to tmp or dev shm",
    "mv":      "move or rename files, file relocation",
    "rm":      "remove files or directories, deletion and cleanup including evidence removal",
    "mkdir":   "create directories, directory creation for staging or persistence",
    "ln":      "create hard or symbolic links, file system manipulation",

    # ── 系统信息类 ──
    "uname":   "print system information, OS and kernel version discovery",
    "hostname":"show or set system hostname, system identification",
    "whoami":  "print current user name, identity verification",
    "id":      "print user and group IDs, identity and privilege enumeration",
    "who":     "show who is logged in, user session reconnaissance",
    "env":     "show or set environment variables, environment inspection",
    "date":    "display or set date and time, system time check",
    "uptime":  "show system uptime and load, system health reconnaissance",
    "free":    "display memory usage, resource reconnaissance",
    "df":      "report disk space usage, storage reconnaissance",
    "du":      "estimate file space usage, storage analysis",
    "pwd":     "print working directory, location check",
    "ls":      "list directory contents, file system enumeration",
    "echo":    "display text, often used in scripts or writing to files",
    "wc":      "count lines words and bytes, data measurement",

    # ── 日志类 ──
    "journalctl": "query systemd journal, system log reconnaissance and monitoring",
    "dmesg":   "print kernel ring buffer, kernel message reconnaissance",
    "last":    "show last logged in users, login history reconnaissance",

    # ── 解析/处理类 ──
    "grep":    "search text patterns in files, data search and filtering",
    "awk":     "pattern scanning and processing language, data extraction",
    "sed":     "stream editor for filtering and transforming text, data modification",
    "sort":    "sort lines of text files, data organization",
    "cut":     "remove sections from lines, data extraction",
    "tr":      "translate or delete characters, data transformation",
    "readlink":"print value of symbolic link, path resolution",
    "dirname": "strip last component from file name, path manipulation",
    "md5sum":  "compute MD5 checksums, file integrity verification",

    # ── 系统后台类 ──
    "suricata": "network IDS IPS engine, intrusion detection monitoring daemon",
    "snap":    "snap package manager, containerized application management",
    "snapctl": "snap control utility, snap configuration",
    "apt":     "advanced package tool, software package management",
    "dpkg":    "debian package manager, package installation and management",
    "getent":  "query name service switch, database lookup utility",
    "modprobe": "load or remove kernel modules, kernel extension management",
    "unix_chkpwd": "verify PAM authentication tokens, password checking utility",

    # ── 编程语言类 ──
    "python3": "Python 3 interpreter, script execution including potential reverse shell or automation",
    "python":  "Python interpreter, script execution and automation",
    "perl":    "Perl interpreter, text processing and scripting including network operations",
    "ruby":    "Ruby interpreter, scripting language execution",
    "node":    "Node.js JavaScript runtime, server-side script execution",
    "bash":    "Bourne again shell, command interpreter and scripting",
    "sh":      "POSIX shell, basic command interpreter",
    "dash":    "Debian almquist shell, lightweight command interpreter",

    # ── 版本控制 ──
    "git":     "distributed version control system, repository management",
    "vim":     "Vi improved text editor, file editing",
}

# ═══ 槽位语义（每个维度的含义描述）═══
SLOT_DESCRIPTIONS = {
    "ET:EXEC": "process execution event",
    "ET:CONN": "network connection event",
    "UID:0":   "root user privileged execution",
    "UID:1000":"normal user execution",
    "DST:EXT:HIGH": "external high-risk network destination",
    "DST:NONE":  "no network destination",
    "DST:OTHER": "other network destination",
    "PC:ETC_PASSWD": "sensitive credential file path",
    "PC:ETC_CRON":   "scheduled task configuration path",
    "PC:ETC_SYSTEMD":"systemd service configuration path",
    "PC:SSH_KEYS":   "SSH key material path",
    "PC:TMP":        "temporary or staging directory path",
    "PC:HOME_RC":    "user shell configuration file path",
    "PC:NONE":       "no special path category",
    "PC:OTHER":      "other filesystem path",
    "PC:ETC_LD":     "dynamic linker configuration path",
    "PC:VAR_LOG":    "system log file path",
}

# ═══ PARENT 语义 ═══
PARENT_DESCRIPTIONS = {
    "bash": "interactive bash shell parent process",
    "sh":   "shell script parent process",
    "sudo": "privileged execution via sudo parent",
    "systemd": "system service manager parent process",
    "python": "Python interpreter parent process",
    "python3": "Python3 interpreter parent process",
    "?":    "unknown or unresolvable parent process",
}


def get_description(token):
    """获取 token 的语义描述"""
    if not ":" in token:
        # DT 桶等无冒号的 token
        return f"time interval bucket {token}"

    slot, val = token.split(":", 1)

    if slot == "PROC":
        return PROC_DESCRIPTIONS.get(val, f"process execution of {val}")
    elif slot in ("ET", "UID", "DST", "PC"):
        return SLOT_DESCRIPTIONS.get(token, f"{slot} value {val}")
    elif slot == "PARENT":
        return PARENT_DESCRIPTIONS.get(val, f"parent process {val}")
    elif slot == "ARGV":
        return f"argument skeleton pattern {val}"
    elif slot == "DT":
        return f"inter-event timing bucket {val}"
    else:
        return f"token {token}"


def build_description_corpus():
    """构建完整描述语料库，返回 {token: description}"""
    corpus = {}

    # PROC 描述
    for proc, desc in PROC_DESCRIPTIONS.items():
        corpus[f"PROC:{proc}"] = desc

    # 槽位描述
    for token, desc in SLOT_DESCRIPTIONS.items():
        corpus[token] = desc

    # PARENT 描述
    for parent, desc in PARENT_DESCRIPTIONS.items():
        corpus[f"PARENT:{parent}"] = desc

    return corpus


if __name__ == "__main__":
    corpus = build_description_corpus()
    print(f"描述库: {len(corpus)} tokens")
    print(f"\n样本:")
    for token in ["PROC:cat", "PROC:ss", "PROC:crontab", "PROC:sudo", "PROC:curl"]:
        print(f"  {token:20s} → {corpus.get(token, '?')[:80]}")
