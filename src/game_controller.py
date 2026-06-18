import os
import time
from typing import Optional

import cv2
import numpy as np

from src.action_executor import ActionExecutor
from src.debug_ui import DebugUI
from src.decision_maker import DecisionMaker
from src.logger import HsBatLogger
from src.state_recognizer import GameState, StateRecognizer


class GameController:
    def __init__(self, config: dict, debug_ui: Optional[DebugUI] = None):
        self.cfg = config
        self._debug_ui = debug_ui
        self.logger = HsBatLogger().get_logger("GameController")
        self.recognizer = StateRecognizer(config)
        self.decision_maker = DecisionMaker(config)
        self.executor = ActionExecutor(config)

        self.debug_cfg = config["debug"]
        self.screenshot_dir = self.debug_cfg.get("screenshot_dir", "screenshots")
        if self.debug_cfg.get("save_screenshots", False):
            os.makedirs(self.screenshot_dir, exist_ok=True)

        self.running = False
        self.paused = False
        self.turn_count = 0
        self.last_state: Optional[GameState] = None

    def _update_debug_ui(self, game_state: GameState):
        if self._debug_ui:
            self._debug_ui.update_state(game_state)

    def run(self):
        self.running = True
        self.logger.info("=" * 50)
        self.logger.info("HsBat 炉石传说自动化脚本 启动")
        self.logger.info("Failsafe: 鼠标移动到左上角(0,0)可紧急停止")
        self.logger.info("=" * 50)

        try:
            while self.running:
                if self.paused:
                    time.sleep(0.5)
                    continue

                self._game_tick()

        except KeyboardInterrupt:
            self.logger.info("用户中断")
        except Exception as e:
            self.logger.error(f"游戏循环异常: {e}", exc_info=True)
        finally:
            self.running = False
            self.logger.info("HsBat 已停止")

    def _game_tick(self):
        game_state = self.recognizer.recognize()
        self.last_state = game_state
        self._update_debug_ui(game_state)

        if self.debug_cfg.get("save_screenshots", False) and game_state.screenshot is not None:
            self._save_screenshot(game_state)

        if game_state.is_game_over:
            self.logger.info("游戏结束")
            time.sleep(5)
            return

        if not game_state.is_our_turn:
            self._wait_for_our_turn(game_state)
            return

        self.turn_count += 1
        self.logger.info(f"\n=== 第 {self.turn_count} 回合 ===")

        self._execute_turn(game_state)

    def _wait_for_our_turn(self, game_state: GameState):
        wait_start = time.time()
        max_wait = 90.0
        check_interval = 1.0

        self.logger.info("等待我方回合...")
        while time.time() - wait_start < max_wait:
            if not self.running or self.paused:
                return

            new_state = self.recognizer.recognize()
            self._update_debug_ui(new_state)
            if new_state.is_our_turn:
                self.logger.info("检测到我方回合开始")
                return
            time.sleep(check_interval)

        self.logger.warning("等待超时，重新检查状态")

    def _execute_turn(self, game_state: GameState):
        max_actions = 20
        action_count = 0

        while action_count < max_actions:
            if not self.running or self.paused:
                return

            current_state = self.recognizer.recognize()
            self._update_debug_ui(current_state)
            if not current_state.is_our_turn:
                self.logger.info("回合已结束")
                return

            if not current_state.has_end_turn_button:
                self.logger.debug("动画播放中，等待...")
                time.sleep(0.5)
                continue

            decision = self.decision_maker.decide(current_state)
            action = decision.get("action", "end_turn")

            self.logger.info(f"执行动作: {action} - {decision.get('reason', '')}")

            if action == "end_turn":
                self._perform_end_turn(current_state)
                return

            elif action == "play_card":
                card_idx = decision.get("card_index")
                if card_idx is not None and card_idx < len(current_state.hand_cards):
                    self._perform_play_card(current_state, card_idx, decision)
                    time.sleep(1.0)

            elif action == "attack":
                self._perform_attacks(current_state)

            elif action == "wait":
                time.sleep(0.5)

            action_count += 1

        self.logger.warning("达到最大动作数，强制结束回合")
        self._perform_end_turn(game_state)

    def _perform_end_turn(self, game_state: GameState):
        self.logger.info("结束回合")
        self.executor.end_turn()
        time.sleep(0.5)

    def _perform_play_card(self, game_state: GameState, card_idx: int, decision: dict):
        if card_idx >= len(game_state.hand_cards):
            self.logger.warning(f"卡牌索引 {card_idx} 超出范围")
            return

        card = game_state.hand_cards[card_idx]
        self.logger.info(f"出牌: [{card.name}] (费用:{card.cost})")

        target_bbox = None
        target_type = decision.get("target_type")

        if target_type == "enemy_hero":
            screen_w = self.cfg["screen"]["game_region"]["width"]
            target_bbox = (screen_w // 2 - 30, 0, screen_w // 2 + 30, 80)
        elif target_type == "enemy_minion":
            target_idx = decision.get("target_index")
            if target_idx is not None and target_idx < len(game_state.opponent_minions):
                target_bbox = game_state.opponent_minions[target_idx].position
        elif target_type == "friendly_minion":
            target_idx = decision.get("target_index")
            if target_idx is not None and target_idx < len(game_state.our_minions):
                target_bbox = game_state.our_minions[target_idx].position

        self.executor.play_card(card.position, target_bbox)

    def _perform_attacks(self, game_state: GameState):
        attack_orders = self.decision_maker.build_attack_plan(game_state)
        for order in attack_orders:
            if not self.running or self.paused:
                return

            attacker_idx = order["attacker_index"]
            if attacker_idx >= len(game_state.our_minions):
                continue

            attacker = game_state.our_minions[attacker_idx]
            target_bbox = None

            if order["target_type"] == "enemy_minion" and order["target_index"] is not None:
                t_idx = order["target_index"]
                if t_idx < len(game_state.opponent_minions):
                    target_bbox = game_state.opponent_minions[t_idx].position

            self.logger.info(f"随从 {attacker_idx} ({attacker.attack}/{attacker.health}) 攻击")
            self.executor.attack_with_minion(attacker.position, target_bbox)
            time.sleep(0.8)

            current_state = self.recognizer.recognize()
            if not current_state.is_our_turn:
                self.logger.info("攻击后回合结束")
                return

    def _save_screenshot(self, game_state: GameState):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"state_{timestamp}_{self.turn_count}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(filepath, game_state.screenshot)
        except Exception as e:
            self.logger.error(f"保存截屏失败: {e}")

    def get_last_state(self) -> Optional[GameState]:
        return self.last_state

    def toggle_pause(self):
        self.paused = not self.paused
        self.logger.info(f"{'暂停' if self.paused else '继续'}执行")

    def stop(self):
        self.running = False
