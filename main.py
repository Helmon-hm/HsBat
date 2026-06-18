#!/usr/bin/env python3
"""
HsBat - 炉石传说混合架构自动化脚本
传统CV快速识别 + 大模型高层决策

支持 GUI 界面和命令行两种模式。
"""

import argparse
import os
import signal
import sys

import yaml

from src.game_controller import GameController
from src.logger import HsBatLogger
from src.paths import get_project_root, get_resource_path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HsBat - 炉石传说自动化脚本 (CV + LLM 混合架构)"
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(get_project_root(), "config.yaml"),
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="启用调试模式",
    )
    parser.add_argument(
        "--no-debug-ui",
        action="store_true",
        help="禁用调试窗口",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="仅使用规则引擎，不调用大模型",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅运行识别模式，不执行动作",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用命令行模式 (默认启动 GUI)",
    )
    return parser.parse_args()


def run_cli(config: dict):
    logger = HsBatLogger(
        log_dir="logs",
        log_level=config["debug"].get("log_level", "DEBUG"),
    ).get_logger("Main")
    logger.info("HsBat v2.0 CLI 模式启动")
    logger.info(f"调试模式: {'开启' if config['debug']['enabled'] else '关闭'}")
    logger.info(f"大模型决策: {'开启' if config['llm']['enabled'] else '关闭 (规则引擎)'}")

    if config["llm"]["enabled"] and not config["llm"].get("api_key"):
        env_key = os.environ.get("HSBAT_LLM_API_KEY", "")
        if env_key:
            config["llm"]["api_key"] = env_key
            logger.info("从环境变量 HSBAT_LLM_API_KEY 读取 API Key")
        else:
            logger.warning("大模型已启用但未配置 API Key，回退到规则引擎")
            config["llm"]["enabled"] = False

    controller = GameController(config)

    def signal_handler(sig, frame):
        logger.info("收到中断信号，正在停止...")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        controller.run()
    finally:
        logger.info("HsBat 已退出")


def find_config(path: str) -> str:
    if os.path.exists(path):
        return path
    bundled = get_resource_path(os.path.basename(path) if os.path.basename(path) else "config.yaml")
    if os.path.exists(bundled):
        print(f"从内建默认配置加载: {bundled}")
        import shutil
        shutil.copy2(bundled, path)
        return path
    return path


def main():
    args = parse_args()
    config_path = find_config(args.config)
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {args.config}")
        print("提示: 首次运行会自动从内建配置生成 config.yaml")
        sys.exit(1)

    config = load_config(config_path)

    if args.debug:
        config["debug"]["enabled"] = True
    if args.no_debug_ui:
        config["debug"]["show_debug_window"] = False
    if args.rule_only:
        config["llm"]["enabled"] = False
    if args.dry_run:
        config["game"]["dry_run"] = True

    if args.cli:
        run_cli(config)
    else:
        from src.gui_app import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
