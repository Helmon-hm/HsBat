import os
import sys
import subprocess


LOG_CONFIG_CONTENT = """[Power]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[Zone]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[Asset]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[Bob]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[Arena]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[LoadingScreen]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[GameState]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false

[Rachelle]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false
"""

_hearthstone_dir_cache = None
_power_log_path_cache = None
_log_config_path_cache = None


def _detect_hearthstone_install_dir():
    """
    Detect Hearthstone installation directory by trying multiple methods.
    Returns the game root directory (where Hearthstone.exe is), or None.
    """
    # 1) Check if Hearthstone process is running
    try:
        result = subprocess.check_output(
            ["wmic", "process", "where", "name='Hearthstone.exe'", "get", "ExecutablePath"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
        for line in result.splitlines():
            line = line.strip()
            if line.lower().endswith("hearthstone.exe"):
                exe_path = line
                install_dir = os.path.dirname(exe_path)
                if os.path.isdir(install_dir):
                    return install_dir
    except Exception:
        pass

    # 2) Check registry (Blizzard/Battle.net installations)
    reg_paths = [
        (0x80000002, r"SOFTWARE\Blizzard Entertainment\Hearthstone"),  # HKLM 32-bit
        (0x80000002, r"SOFTWARE\WOW6432Node\Blizzard Entertainment\Hearthstone"),
        (0x80000001, r"SOFTWARE\Blizzard Entertainment\Hearthstone"),  # HKCU
    ]
    import winreg
    for hive, subkey in reg_paths:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                path, _ = winreg.QueryValueEx(key, "InstallPath")
                if path and os.path.isdir(path):
                    return path
        except OSError:
            continue

    # 3) Scan common installation directories
    common_paths = []
    for drive in ["C:", "D:", "E:", "F:", "G:"]:
        for sub in [
            r"\Program Files (x86)\Hearthstone",
            r"\Program Files\Hearthstone",
            r"\Games\Hearthstone",
            r"\Battle.net\Hearthstone",
            r"\Blizzard\Hearthstone",
            r"\炉石传说",
        ]:
            common_paths.append(drive + sub)

    # Also check AppData paths
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        common_paths.append(os.path.join(local_app_data, "Blizzard", "Hearthstone"))

    app_data = os.environ.get("APPDATA", "")
    if app_data:
        common_paths.append(os.path.join(app_data, "Blizzard", "Hearthstone"))

    program_data = os.environ.get("PROGRAMDATA", "")
    if program_data:
        common_paths.append(os.path.join(program_data, "Blizzard Entertainment", "Hearthstone"))

    for p in common_paths:
        if os.path.isdir(p):
            return p

    return None


def _find_latest_log_subdir(logs_dir):
    """Find the most recent timestamped subdirectory inside Logs/."""
    if not os.path.isdir(logs_dir):
        return None
    candidates = []
    for name in os.listdir(logs_dir):
        sub = os.path.join(logs_dir, name)
        if not os.path.isdir(sub):
            continue
        # Hearthstone creates dirs like "Hearthstone_2026_06_21_16_31_21"
        # or "2025-06-21-16-30-45"
        clean = name.replace("Hearthstone_", "").replace("Hearthstone-", "")
        parts = clean.replace("_", "-").split("-")
        if len(parts) >= 6 and all(p.isdigit() for p in parts[:6]):
            candidates.append((clean, sub))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_power_log_in_dir(game_dir):
    """Search for Power.log in or near the game directory."""
    logs_dir = os.path.join(game_dir, "Logs")

    # Hearthstone writes logs into timestamped subdirectories: Logs/2025-06-21-16-30-45/Power.log
    latest_subdir = _find_latest_log_subdir(logs_dir)
    if latest_subdir:
        power_log = os.path.join(latest_subdir, "Power.log")
        if os.path.isfile(power_log):
            return power_log

    # Direct Logs subdirectory (older HS versions)
    power_log = os.path.join(logs_dir, "Power.log")
    if os.path.isfile(power_log):
        return power_log

    # Some installations write logs to AppData instead
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        appdata_log = os.path.join(local_app_data, "Blizzard", "Hearthstone", "Logs")
        latest_sub = _find_latest_log_subdir(appdata_log)
        if latest_sub:
            pl = os.path.join(latest_sub, "Power.log")
            if os.path.isfile(pl):
                return pl
        appdata_log = os.path.join(appdata_log, "Power.log")
        if os.path.isfile(appdata_log):
            return appdata_log

    app_data = os.environ.get("APPDATA", "")
    if app_data:
        appdata_log = os.path.join(app_data, "Blizzard", "Hearthstone", "Logs")
        latest_sub = _find_latest_log_subdir(appdata_log)
        if latest_sub:
            pl = os.path.join(latest_sub, "Power.log")
            if os.path.isfile(pl):
                return pl

    # Check if game_dir itself contains Power.log (unusual but possible)
    direct = os.path.join(game_dir, "Power.log")
    if os.path.isfile(direct):
        return direct

    # Default: assume Logs subdirectory (will be created by game)
    return os.path.join(logs_dir, "Power.log")


def get_hearthstone_game_dir():
    """Get detected Hearthstone installation directory."""
    global _hearthstone_dir_cache
    if _hearthstone_dir_cache is None:
        _hearthstone_dir_cache = _detect_hearthstone_install_dir()
    return _hearthstone_dir_cache


def get_hearthstone_log_dir():
    """Get the directory where Power.log is (or will be) written."""
    game_dir = get_hearthstone_game_dir()
    if game_dir:
        logs_dir = os.path.join(game_dir, "Logs")
        if os.path.isdir(logs_dir):
            latest = _find_latest_log_subdir(logs_dir)
            if latest:
                return latest
            return logs_dir
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            appdata_logs = os.path.join(local_app_data, "Blizzard", "Hearthstone", "Logs")
            if os.path.isdir(appdata_logs):
                latest = _find_latest_log_subdir(appdata_logs)
                if latest:
                    return latest
                return appdata_logs
        return logs_dir
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if not local_app_data:
        home = os.path.expanduser("~")
        local_app_data = os.path.join(home, "AppData", "Local")
    return os.path.join(local_app_data, "Blizzard", "Hearthstone", "Logs")


def get_log_config_path():
    """Get the path where log.config should be placed."""
    global _log_config_path_cache
    if _log_config_path_cache is not None:
        return _log_config_path_cache

    game_dir = get_hearthstone_game_dir()
    if game_dir:
        config_path = os.path.join(game_dir, "log.config")
        if os.path.isfile(config_path):
            _log_config_path_cache = config_path
            return config_path

    # Also check AppData locations
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    appdata_config = os.path.join(local_app_data, "Blizzard", "Hearthstone", "log.config")
    if os.path.isfile(appdata_config):
        _log_config_path_cache = appdata_config
        return appdata_config

    # Prefer game directory for new config files
    if game_dir:
        _log_config_path_cache = os.path.join(game_dir, "log.config")
    elif local_app_data:
        _log_config_path_cache = os.path.join(local_app_data, "Blizzard", "Hearthstone", "log.config")
    else:
        home = os.path.expanduser("~")
        _log_config_path_cache = os.path.join(home, "AppData", "Local", "Blizzard", "Hearthstone", "log.config")

    return _log_config_path_cache


def get_power_log_path(manual_path=None):
    """
    Get the path to Power.log.
    priority: manual config > process detection > AppData scan > default
    """
    global _power_log_path_cache

    if manual_path and os.path.isfile(manual_path):
        return manual_path
    if manual_path and os.path.isdir(manual_path):
        return os.path.join(manual_path, "Power.log")

    if _power_log_path_cache is not None:
        return _power_log_path_cache

    game_dir = get_hearthstone_game_dir()
    if game_dir:
        _power_log_path_cache = _find_power_log_in_dir(game_dir)
        return _power_log_path_cache

    # Scan all possible locations for Power.log
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        p = os.path.join(local_app_data, "Blizzard", "Hearthstone", "Logs", "Power.log")
        if os.path.isfile(p):
            _power_log_path_cache = p
            return p

    app_data = os.environ.get("APPDATA", "")
    if app_data:
        p = os.path.join(app_data, "Blizzard", "Hearthstone", "Logs", "Power.log")
        if os.path.isfile(p):
            _power_log_path_cache = p
            return p

    # Scan common drives for Power.log (including timestamped subdirs)
    for drive in ["C:", "D:", "E:", "F:", "G:"]:
        for sub in [
            r"\Program Files (x86)\Hearthstone\Logs",
            r"\Program Files\Hearthstone\Logs",
            r"\Games\Hearthstone\Logs",
            r"\炉石传说\Logs",
        ]:
            logs_dir = drive + sub
            if os.path.isdir(logs_dir):
                latest = _find_latest_log_subdir(logs_dir)
                if latest:
                    p = os.path.join(latest, "Power.log")
                    if os.path.isfile(p):
                        _power_log_path_cache = p
                        return p

    # Default fallback
    _power_log_path_cache = os.path.join(get_hearthstone_log_dir(), "Power.log")
    return _power_log_path_cache


def is_log_config_valid():
    config_path = get_log_config_path()
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        return "[Power]" in content and "LogLevel=1" in content and "FilePrinting=true" in content
    except Exception:
        return False


def ensure_log_config(force=False):
    config_path = get_log_config_path()
    if is_log_config_valid() and not force:
        return True, "log.config 已存在且有效"

    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(LOG_CONFIG_CONTENT)
        return True, f"log.config 已创建: {config_path}"
    except PermissionError:
        return False, f"权限不足，无法创建 log.config: {config_path}"
    except Exception as e:
        return False, f"创建 log.config 失败: {e}"


def check_log_availability():
    game_dir = get_hearthstone_game_dir()
    config_ok = is_log_config_valid()
    log_path = get_power_log_path()
    log_exists = os.path.isfile(log_path)
    log_dir = os.path.dirname(log_path)
    log_dir_exists = os.path.isdir(log_dir)

    issues = []
    if not game_dir:
        issues.append("未检测到炉石传说安装目录（进程/注册表/常见路径均未找到）")
    if not config_ok:
        issues.append(f"log.config 缺失或无效")
    if not log_dir_exists:
        issues.append(f"日志目录不存在: {log_dir}")

    if issues:
        return False, issues
    return True, []


def reset_cache():
    """Reset cached paths (call after user changes config or HS is restarted)."""
    global _hearthstone_dir_cache, _power_log_path_cache, _log_config_path_cache
    _hearthstone_dir_cache = None
    _power_log_path_cache = None
    _log_config_path_cache = None
