import logging
import os
import queue
import threading
import time
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter as tk

import yaml

from src.game_controller import GameController
from src.logger import HsBatLogger
from src.paths import get_project_root, get_resource_path

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False


class GuiLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setLevel(logging.DEBUG)

    def emit(self, record):
        self.log_queue.put(record)


class HsBatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HsBat - 炉石传说自动化")
        self.root.geometry("960x1050")
        self.root.minsize(820, 800)
        self._tray_icon = None
        self._minimize_to_tray = False

        self.config_path = os.path.join(get_project_root(), "config.yaml")
        self.config = self._load_config()

        self.controller: GameController = None
        self.controller_thread: threading.Thread = None
        self.bot_running = False

        self.log_queue = queue.Queue()
        self._setup_log_redirect()

        self._build_ui()
        self._load_config_to_ui()
        self._poll_log_queue()

    def _get_screen_size(self):
        try:
            import pyautogui
            return pyautogui.size()
        except Exception:
            return (1920, 1080)

    def _auto_detect_screen(self, cfg: dict):
        region = cfg.setdefault("screen", {}).setdefault("game_region", {})
        if region.get("width", 0) <= 0 or region.get("height", 0) <= 0:
            sw, sh = self._get_screen_size()
            region["width"] = sw
            region["height"] = sh

    def _load_config(self) -> dict:
        cfg_path = self.config_path
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            bundled = get_resource_path("config.yaml")
            if os.path.exists(bundled):
                with open(bundled, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)
            else:
                cfg = {}
        self._auto_detect_screen(cfg)
        return cfg

    def _save_config(self):
        path = self.config_path
        self._ui_to_config()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)

    def _setup_log_redirect(self):
        logger_obj = HsBatLogger(log_dir="logs", log_level="DEBUG")
        handler = GuiLogHandler(self.log_queue)
        logger_obj.logger.addHandler(handler)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_control_tab(notebook)
        self._build_settings_tab(notebook)
        self._build_log_tab(notebook)

    # ---------- 控制标签页 (第一个, 最常用) ----------
    def _build_control_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="控制")

        ctrl_frame = ttk.LabelFrame(tab, text="Bot 控制", padding=10)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(ctrl_frame, text="启动 Bot",
                                     command=self._start_bot, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_btn = ttk.Button(ctrl_frame, text="停止 Bot",
                                    command=self._stop_bot, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = ttk.Label(ctrl_frame, text="状态: 未启动",
                                       font=("", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))

        # ---------- 决策模式 (一目了然) ----------
        mode_frame = ttk.LabelFrame(tab, text="决策模式 — 选一种", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.decision_mode_var = tk.StringVar(value="llm")
        rb_llm = ttk.Radiobutton(
            mode_frame, text="智能模式 (大模型 AI 决策)",
            variable=self.decision_mode_var, value="llm",
            command=self._on_decision_mode_change,
        )
        rb_llm.pack(anchor="w", pady=2)
        ttk.Label(mode_frame,
                  text="     调用大模型 API 分析局势后做出最优决策（需配置 API Key）",
                  foreground="#888888").pack(anchor="w")

        rb_rule = ttk.Radiobutton(
            mode_frame, text="快速模式 (规则引擎决策)",
            variable=self.decision_mode_var, value="rule",
            command=self._on_decision_mode_change,
        )
        rb_rule.pack(anchor="w", pady=2)
        ttk.Label(mode_frame,
                  text="     使用内置规则：有费出牌 → 随从攻击 → 结束回合（零延迟，无需网络）",
                  foreground="#888888").pack(anchor="w")

        self.mode_hint_label = ttk.Label(mode_frame, text="",
                                          foreground="#d4a017")
        self.mode_hint_label.pack(anchor="w", pady=(5, 0))

        # ---------- 辅助选项 ----------
        opt_frame = ttk.LabelFrame(tab, text="辅助选项", padding=10)
        opt_frame.pack(fill=tk.X, pady=(0, 10))

        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="模拟运行 (只识别不动手，适合测试识别效果)",
                         variable=self.dry_run_var).pack(anchor="w", pady=1)
        self.auto_requeue_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="对局结束后自动开始下一局",
                         variable=self.auto_requeue_var).pack(anchor="w", pady=1)

        # ---------- 当前游戏状态 ----------
        state_frame = ttk.LabelFrame(tab, text="当前游戏状态", padding=10)
        state_frame.pack(fill=tk.BOTH, expand=True)

        self.state_text = tk.Text(state_frame, height=8, wrap=tk.WORD,
                                   font=("Consolas", 10),
                                   state=tk.DISABLED,
                                   bg="#1e1e1e", fg="#d4d4d4",
                                   insertbackground="white")
        self.state_text.pack(fill=tk.BOTH, expand=True)

    def _on_decision_mode_change(self):
        mode = self.decision_mode_var.get()
        llm_cfg = self.config.get("llm", {})
        if mode == "llm":
            if not llm_cfg.get("api_key") and not os.environ.get("HSBAT_LLM_API_KEY", ""):
                self.mode_hint_label.config(
                    text="提示：未配置 API Key，启动后将自动回退到规则引擎",
                    foreground="#d4a017")
            else:
                self.mode_hint_label.config(text="")
        else:
            self.mode_hint_label.config(text="")

    # ---------- 设置标签页 ----------
    def _build_settings_tab(self, notebook):
        outer = ttk.Frame(notebook)
        notebook.add(outer, text="设置")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        tab = ttk.Frame(canvas, padding=10)

        tab.columnconfigure(0, weight=1)

        canvas.create_window((0, 0), window=tab, anchor="nw", tags=("inner_win",))

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        tab.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig("inner_win", width=event.width - 4)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+"))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        outer.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        row = 0

        llm_frame = ttk.LabelFrame(tab, text="大模型 (LLM) 配置", padding=10)
        llm_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        llm_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(llm_frame, text="API 地址:")\
            .grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        self.api_base_var = tk.StringVar()
        ttk.Entry(llm_frame, textvariable=self.api_base_var, width=60)\
            .grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(llm_frame, text="API Key:")\
            .grid(row=2, column=0, sticky="w", padx=(0, 5), pady=2)
        self.api_key_var = tk.StringVar()
        key_frame = ttk.Frame(llm_frame)
        key_frame.grid(row=2, column=1, sticky="ew", pady=2)
        key_frame.columnconfigure(0, weight=1)
        self.api_key_entry = ttk.Entry(key_frame, textvariable=self.api_key_var, width=60, show="*")
        self.api_key_entry.grid(row=0, column=0, sticky="ew")
        self._show_key_btn = ttk.Button(key_frame, text="显示", width=6, command=self._toggle_key_visibility)
        self._show_key_btn.grid(row=0, column=1, padx=(5, 0))
        self._key_visible = False

        ttk.Label(llm_frame, text="模型名称:")\
            .grid(row=3, column=0, sticky="w", padx=(0, 5), pady=2)
        self.model_var = tk.StringVar()
        ttk.Combobox(llm_frame, textvariable=self.model_var, values=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "deepseek-chat", "qwen2.5-72b-instruct",
        ], width=57).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(llm_frame, text="Temperature:")\
            .grid(row=4, column=0, sticky="w", padx=(0, 5), pady=2)
        self.temperature_var = tk.DoubleVar(value=0.3)
        temp_frame = ttk.Frame(llm_frame)
        temp_frame.grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Scale(temp_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                   variable=self.temperature_var, length=300)\
            .pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.temperature_label = ttk.Label(temp_frame, text="0.3", width=4)
        self.temperature_label.pack(side=tk.LEFT, padx=(5, 0))
        self.temperature_var.trace_add(
            "write",
            lambda *_: self.temperature_label.config(text=f"{self.temperature_var.get():.1f}"),
        )

        row += 1
        game_frame = ttk.LabelFrame(tab, text="游戏与识别", padding=10)
        game_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        game_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(game_frame, text="屏幕宽度:")\
            .grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        self.screen_w_var = tk.IntVar(value=1920)
        ttk.Spinbox(game_frame, from_=800, to=3840, textvariable=self.screen_w_var, width=10)\
            .grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(game_frame, text="屏幕高度:")\
            .grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        self.screen_h_var = tk.IntVar(value=1080)
        ttk.Spinbox(game_frame, from_=600, to=2160, textvariable=self.screen_h_var, width=10)\
            .grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(game_frame, text="Tesseract 路径:")\
            .grid(row=2, column=0, sticky="w", padx=(0, 5), pady=2)
        tess_frame = ttk.Frame(game_frame)
        tess_frame.grid(row=2, column=1, sticky="ew", pady=2)
        tess_frame.columnconfigure(0, weight=1)
        self.tesseract_path_var = tk.StringVar()
        ttk.Entry(tess_frame, textvariable=self.tesseract_path_var)\
            .grid(row=0, column=0, sticky="ew")
        ttk.Button(tess_frame, text="浏览", width=6, command=self._browse_tesseract)\
            .grid(row=0, column=1, padx=(5, 0))

        row += 1
        # ---------- 规则引擎策略 ----------
        rule_frame = ttk.LabelFrame(tab, text="规则引擎策略（快速模式）", padding=10)
        rule_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        rule_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(rule_frame, text="出牌策略:")\
            .grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        self.rule_play_var = tk.StringVar(value="high_cost_first")
        play_frame = ttk.Frame(rule_frame)
        play_frame.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Radiobutton(play_frame, text="优先出高费",
                         variable=self.rule_play_var, value="high_cost_first")\
            .pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(play_frame, text="优先出低费 (铺场)",
                         variable=self.rule_play_var, value="low_cost_first")\
            .pack(side=tk.LEFT)

        ttk.Label(rule_frame, text="攻击策略:")\
            .grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        self.rule_attack_var = tk.StringVar(value="smart")
        attack_frame = ttk.Frame(rule_frame)
        attack_frame.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Radiobutton(attack_frame, text="智能", variable=self.rule_attack_var,
                         value="smart").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(attack_frame, text="只打脸", variable=self.rule_attack_var,
                         value="face_only").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(attack_frame, text="只解场", variable=self.rule_attack_var,
                         value="trade_only").pack(side=tk.LEFT)

        ttk.Label(rule_frame, text="     智能模式: 有斩杀时打脸，有威胁时解场，否则打脸",
                  foreground="#888888")\
            .grid(row=2, column=1, sticky="w")

        self.rule_defend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rule_frame, text="危险时优先解场 (敌方场攻超过斩杀线时)",
                         variable=self.rule_defend_var)\
            .grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))

        ttk.Label(rule_frame, text="斩杀线余量:")\
            .grid(row=4, column=0, sticky="w", padx=(0, 5), pady=2)
        self.rule_lethal_margin_var = tk.IntVar(value=2)
        ttk.Spinbox(rule_frame, from_=0, to=10, textvariable=self.rule_lethal_margin_var, width=6)\
            .grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(rule_frame, text="  超出血量多少点触发解场", foreground="#888888")\
            .grid(row=4, column=1, padx=(45, 0), sticky="w", pady=2)

        self.rule_hero_power_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rule_frame, text="使用英雄技能 (剩余费用 >= 2 时)",
                         variable=self.rule_hero_power_var)\
            .grid(row=5, column=0, columnspan=2, sticky="w", pady=2)

        row += 1
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=row, column=0, pady=10)
        ttk.Button(btn_frame, text="保存配置", command=self._save_config_ui)\
            .pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="恢复默认", command=self._reset_config)\
            .pack(side=tk.LEFT, padx=5)

    # ---------- 日志标签页 ----------
    def _build_log_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="日志")

        self.log_text = scrolledtext.ScrolledText(
            tab, wrap=tk.WORD, font=("Consolas", 9),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white",
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_frame, text="清空日志", command=self._clear_log)\
            .pack(side=tk.LEFT, padx=(0, 5))

    # ---------- 交互回调 ----------
    def _toggle_key_visibility(self):
        self._key_visible = not self._key_visible
        self.api_key_entry.config(show="" if self._key_visible else "*")
        self._show_key_btn.config(text="隐藏" if self._key_visible else "显示")

    def _browse_tesseract(self):
        path = filedialog.askopenfilename(
            title="选择 tesseract.exe",
            filetypes=[("可执行文件", "*.exe")],
        )
        if path:
            self.tesseract_path_var.set(path)

    def _load_config_to_ui(self):
        cfg = self.config
        llm = cfg.get("llm", {})
        self.api_base_var.set(llm.get("api_base", "https://api.openai.com/v1"))
        self.api_key_var.set(llm.get("api_key", ""))
        self.model_var.set(llm.get("model", "gpt-4o"))
        self.temperature_var.set(llm.get("temperature", 0.3))
        self.temperature_label.config(text=f"{self.temperature_var.get():.1f}")

        llm_enabled = llm.get("enabled", True)
        self.decision_mode_var.set("llm" if llm_enabled else "rule")
        self._on_decision_mode_change()

        screen = cfg.get("screen", {}).get("game_region", {})
        sw, sh = self._get_screen_size()
        self.screen_w_var.set(screen.get("width", sw) or sw)
        self.screen_h_var.set(screen.get("height", sh) or sh)

        ocr = cfg.get("ocr", {})
        self.tesseract_path_var.set(ocr.get("tesseract_path", ""))

        rule = cfg.get("rule_engine", {})
        self.rule_play_var.set(rule.get("play_card_strategy", "high_cost_first"))
        self.rule_attack_var.set(rule.get("attack_strategy", "smart"))
        self.rule_defend_var.set(rule.get("defend_when_lethal", True))
        self.rule_lethal_margin_var.set(rule.get("lethal_margin", 2))
        self.rule_hero_power_var.set(rule.get("use_hero_power", True))

        game_cfg = cfg.get("game", {})
        self.auto_requeue_var.set(game_cfg.get("auto_requeue", True))

    def _ui_to_config(self):
        self.config.setdefault("llm", {})["enabled"] = (self.decision_mode_var.get() == "llm")
        self.config["llm"]["api_base"] = self.api_base_var.get().strip()
        self.config["llm"]["api_key"] = self.api_key_var.get().strip()
        self.config["llm"]["model"] = self.model_var.get().strip()
        self.config["llm"]["temperature"] = round(self.temperature_var.get(), 1)
        self.config.setdefault("screen", {}).setdefault("game_region", {})["width"] = self.screen_w_var.get()
        self.config["screen"]["game_region"]["height"] = self.screen_h_var.get()
        self.config.setdefault("ocr", {})["tesseract_path"] = self.tesseract_path_var.get().strip()
        self.config.setdefault("rule_engine", {})["play_card_strategy"] = self.rule_play_var.get()
        self.config["rule_engine"]["attack_strategy"] = self.rule_attack_var.get()
        self.config["rule_engine"]["defend_when_lethal"] = self.rule_defend_var.get()
        self.config["rule_engine"]["lethal_margin"] = self.rule_lethal_margin_var.get()
        self.config["rule_engine"]["use_hero_power"] = self.rule_hero_power_var.get()
        self.config.setdefault("game", {})["auto_requeue"] = self.auto_requeue_var.get()

    def _save_config_ui(self):
        self._ui_to_config()
        try:
            self._save_config()
            messagebox.showinfo("保存成功", "配置已保存到 config.yaml")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _reset_config(self):
        if messagebox.askyesno("确认", "恢复默认配置?"):
            self.decision_mode_var.set("llm")
            self.api_base_var.set("https://api.openai.com/v1")
            self.api_key_var.set("")
            self.model_var.set("gpt-4o")
            self.temperature_var.set(0.3)
            sw, sh = self._get_screen_size()
            self.screen_w_var.set(sw)
            self.screen_h_var.set(sh)
            self.tesseract_path_var.set("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            self.temperature_label.config(text="0.3")
            self.rule_play_var.set("high_cost_first")
            self.rule_attack_var.set("smart")
            self.rule_defend_var.set(True)
            self.rule_lethal_margin_var.set(2)
            self.rule_hero_power_var.set(True)
            self.auto_requeue_var.set(True)
            self._on_decision_mode_change()

    def _start_bot(self):
        if self.bot_running:
            return

        self._ui_to_config()
        config = self.config

        if config["llm"]["enabled"] and not config["llm"].get("api_key") and not os.environ.get("HSBAT_LLM_API_KEY", ""):
            if not messagebox.askyesno("警告", "大模型已启用但未配置 API Key，将回退到规则引擎。继续?"):
                return

        if config["llm"]["enabled"] and not config["llm"].get("api_key"):
            env_key = os.environ.get("HSBAT_LLM_API_KEY", "")
            if env_key:
                config["llm"]["api_key"] = env_key
            else:
                config["llm"]["enabled"] = False

        if self.dry_run_var.get():
            config.setdefault("game", {})["dry_run"] = True

        mode_name = "大模型决策" if config["llm"]["enabled"] else "规则引擎"
        self._append_log("INFO", "Main", f"决策模式: {mode_name}")
        if config.get("game", {}).get("dry_run"):
            self._append_log("INFO", "Main", "模拟模式: 仅识别，不执行鼠标操作")

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="状态: 运行中...")

        self.controller = GameController(config)
        self.bot_running = True
        self.controller_thread = threading.Thread(target=self._run_controller, daemon=True)
        self.controller_thread.start()

    def _run_controller(self):
        try:
            self._append_log("INFO", "Main", "HsBat 在 GUI 模式下启动")
            self.controller.run()
        except Exception as e:
            self._append_log("ERROR", "Main", f"控制器异常: {e}")
        finally:
            self.bot_running = False
            self.root.after(0, self._on_bot_stopped)

    def _stop_bot(self):
        if self.controller:
            self.controller.stop()
        self.status_label.config(text="状态: 正在停止...")

    def _on_bot_stopped(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止")
        self._append_log("INFO", "Main", "HsBat 已停止")

    def _append_log(self, level, name, message):
        self.log_text.config(state=tk.NORMAL)
        tag = level.lower()
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} [{level}] {name} - {message}\n", tag)
        self.log_text.tag_config("info", foreground="#d4d4d4")
        self.log_text.tag_config("debug", foreground="#569cd6")
        self.log_text.tag_config("warning", foreground="#dcdcaa")
        self.log_text.tag_config("error", foreground="#f44747")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_log_queue(self):
        try:
            while True:
                record = self.log_queue.get_nowait()
                level = getattr(record, "levelname", "INFO")
                name = getattr(record, "name", "")
                msg = getattr(record, "msg", "")
                self._append_log(level, name, str(msg))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_log_queue)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ---------- 系统托盘 ----------
    def _create_tray_icon(self):
        if not HAS_TRAY:
            return
        width = 64
        height = 64
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse([4, 4, width - 4, height - 4], fill=(0, 120, 212, 255))
        draw.text((width // 2 - 12, height // 2 - 10), "HS", fill=(255, 255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem("显示窗口", self._show_window, default=True),
            pystray.MenuItem("启动 Bot" if not self.bot_running else "停止 Bot",
                             self._tray_toggle_bot),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._tray_quit),
        )
        self._tray_icon = pystray.Icon("hsbat", image, "HsBat - 炉石传说自动化", menu)

    def _show_window(self):
        self._minimize_to_tray = False
        self.root.deiconify()
        self.root.lift()

    def _hide_to_tray(self):
        if self._tray_icon is None:
            return
        self._minimize_to_tray = True
        self.root.withdraw()
        if not self._tray_icon.visible:
            threading.Thread(target=self._tray_icon.run, daemon=True).start()

    def _tray_toggle_bot(self):
        if self.bot_running:
            self.root.after(0, self._stop_bot)
        else:
            self.root.after(0, self._start_bot)

    def _tray_quit(self):
        if self.bot_running:
            self._stop_bot()
            time.sleep(0.3)
        if self._tray_icon:
            self._tray_icon.stop()
        self.root.after(0, self.root.destroy)

    # ---------- 生命周期 ----------
    def run(self):
        if HAS_TRAY:
            self._create_tray_icon()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if HAS_TRAY and self._tray_icon is not None:
            self._hide_to_tray()
        else:
            self._really_quit()

    def _really_quit(self):
        if self.bot_running:
            if not messagebox.askyesno("确认退出", "Bot 仍在运行，确定要退出吗?"):
                return
            self._stop_bot()
            time.sleep(0.5)
        try:
            if self._tray_icon:
                self._tray_icon.stop()
            self.root.destroy()
        except Exception:
            pass


def launch_gui():
    app = HsBatGUI()
    app.run()
