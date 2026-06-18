import os
import time
from typing import Optional

import cv2
import numpy as np

from src.action_executor import ActionExecutor
from src.decision_maker import DecisionMaker
from src.logger import HsBatLogger
from src.state_recognizer import GameState, StateRecognizer


class GameController:
    def __init__(self, config: dict):
        self.cfg = config
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
        self.match_count = 0
        self.last_state: Optional[GameState] = None

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

        if self.debug_cfg.get("save_screenshots", False) and game_state.screenshot is not None:
            self._save_screenshot(game_state)

        if game_state.is_game_over or game_state.is_post_game:
            self._handle_post_game(game_state)
            return

        if game_state.is_main_menu:
            self._handle_main_menu()
            return

        if not game_state.is_our_turn:
            self._wait_for_our_turn(game_state)
            return

        self.turn_count += 1
        self.logger.info(f"\n=== 第 {self.turn_count} 回合 ===")

        self._execute_turn(game_state)

    def _handle_post_game(self, game_state: GameState):
        game_cfg = self.cfg.get("game", {})
        if not game_cfg.get("auto_requeue", True):
            self.logger.info("auto_requeue 已关闭，等待手动操作")
            time.sleep(5)
            return

        self.logger.info("对局已结束，正在处理结算画面...")
        self.turn_count = 0

        click_region = game_cfg.get("post_game_click_region", [0.21, 0.60])
        screen_w = self.cfg["screen"]["game_region"]["width"]
        screen_h = self.cfg["screen"]["game_region"]["height"]
        cx = int(click_region[0] * screen_w)
        cy = int(click_region[1] * screen_h)

        max_clicks = 10
        for i in range(max_clicks):
            if not self.running or self.paused:
                return

            state = self.recognizer.recognize()
    
            if state.is_main_menu:
                self.logger.info("已回到主菜单")
                self._handle_main_menu()
                return

            self.logger.info(f"点击结算画面 ({i + 1}/{max_clicks})...")
            self.executor.click_screen_region(cx, cy)
            time.sleep(1.5)

        self.logger.warning("结算画面处理超时，尝试直接查找主菜单")
        self._handle_main_menu()

    def _handle_main_menu(self):
        game_cfg = self.cfg.get("game", {})
        if not game_cfg.get("auto_requeue", True):
            self.logger.info("auto_requeue 已关闭，停留主菜单等待手动操作")
            time.sleep(5)
            return

        self.logger.info("在主菜单，准备开始新对局...")
        time.sleep(2)

        play_region = game_cfg.get("play_button_region", [0.36, 0.56, 0.27, 0.06])
        screen_w = self.cfg["screen"]["game_region"]["width"]
        screen_h = self.cfg["screen"]["game_region"]["height"]
        px = int(play_region[0] * screen_w + play_region[2] * screen_w // 2)
        py = int(play_region[1] * screen_h + play_region[3] * screen_h // 2)

        for attempt in range(8):
            if not self.running or self.paused:
                return

            state = self.recognizer.recognize()
    
            if state.is_our_turn:
                self.logger.info("检测到对局已开始!")
                self.match_count += 1
                self.logger.info(f"开始第 {self.match_count} 局")
                return

            if not state.is_main_menu and not state.is_game_over:
                self.logger.info("检测到非主菜单画面，等待进入对局...")
                time.sleep(2)
                continue

            self.logger.info(f"点击开始按钮 ({attempt + 1}/8)...")
            self.executor.click_screen_region(px, py)
            time.sleep(2)

        state = self.recognizer.recognize()
        if state.is_our_turn:
            self.match_count += 1
            self.logger.info(f"开始第 {self.match_count} 局")
        else:
            self.logger.warning("未能检测到对局开始，稍后重试")
            time.sleep(5)

    def _wait_for_our_turn(self, game_state: GameState):
        wait_start = time.time()
        max_wait = 90.0
        check_interval = 1.0

        self.logger.info("等待我方回合...")
        while time.time() - wait_start < max_wait:
            if not self.running or self.paused:
                return

            new_state = self.recognizer.recognize()
    
            if new_state.is_game_over or new_state.is_post_game:
                self._handle_post_game(new_state)
                return

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
    
            if current_state.is_game_over or current_state.is_post_game:
                self._handle_post_game(current_state)
                return

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
            screen_h = self.cfg["screen"]["game_region"]["height"]
            hx = screen_w // 2 - int(0.016 * screen_w)
            target_bbox = (hx, 0, hx + int(0.031 * screen_w), int(0.074 * screen_h))
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
            if current_state.is_game_over or current_state.is_post_game:
                self._handle_post_game(current_state)
                return
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
