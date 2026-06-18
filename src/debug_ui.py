import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.logger import HsBatLogger
from src.state_recognizer import GameState


class DebugUI:
    def __init__(self, config: dict):
        self.cfg = config
        self.logger = HsBatLogger().get_logger("DebugUI")
        self.enabled = config["debug"].get("show_debug_window", False)
        self.update_interval = config["debug"].get("ui_update_interval", 0.5)
        self.window_name = "HsBat Debug"
        self._game_state: Optional[GameState] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._ui_loop, daemon=True)
        self._thread.start()
        self.logger.info("调试UI已启动")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass

    def update_state(self, game_state: GameState):
        with self._lock:
            self._game_state = game_state

    def _ui_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 600)

        while self._running:
            with self._lock:
                state = self._game_state

            if state is not None and state.screenshot is not None:
                display = self._build_display(state)
                if display is not None:
                    cv2.imshow(self.window_name, display)
            else:
                blank = np.zeros((600, 960, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Waiting for game state...",
                    (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(self.window_name, blank)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.logger.info("调试窗口关闭")
                break
            elif key == ord(" "):
                self.logger.info("暂停/继续 (快捷键)")
            elif key == ord("s"):
                if state and state.screenshot is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"debug_screenshot_{ts}.png", state.screenshot)
                    self.logger.info(f"调试截图已保存: debug_screenshot_{ts}.png")

            time.sleep(self.update_interval)

        cv2.destroyWindow(self.window_name)

    def _build_display(self, state: GameState) -> Optional[np.ndarray]:
        if state.screenshot is None:
            return None

        h, w = state.screenshot.shape[:2]
        scale = min(480 / h, 960 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(state.screenshot, (new_w, new_h))

        overlay = np.zeros((new_h + 180, new_w, 3), dtype=np.uint8)
        overlay[:new_h, :new_w] = display

        info_y = new_h + 10
        info_lines = [
            f"回合: {'我方' if state.is_our_turn else '敌方'} | "
            f"血量: {state.our_health}/{state.opponent_health} | "
            f"法力: {state.our_mana}/{state.total_mana}",
            f"手牌: {len(state.hand_cards)}张 | "
            f"我方随从: {len(state.our_minions)} | "
            f"敌方随从: {len(state.opponent_minions)} | "
            f"回合按钮: {'可见' if state.has_end_turn_button else '隐藏'}",
        ]

        card_text = "手牌: "
        for i, c in enumerate(state.hand_cards):
            card_text += f"[{i}]{c.name}({c.cost}) "
        if len(card_text) > 200:
            card_text = card_text[:200] + "..."

        info_lines.append(card_text)

        minion_text = "我方随从: "
        for i, m in enumerate(state.our_minions):
            minion_text += f"[{i}]{m.attack}/{m.health} "
        info_lines.append(minion_text)

        enemy_text = "敌方随从: "
        for i, m in enumerate(state.opponent_minions):
            enemy_text += f"[{i}]{m.attack}/{m.health} "
        info_lines.append(enemy_text)

        for i, line in enumerate(info_lines):
            y = info_y + i * 25
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        scale_orig = 1.0 / scale
        for card in state.hand_cards:
            x1, y1, x2, y2 = card.position
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            color = (0, 255, 0) if card.is_playable else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        for minion in state.our_minions:
            x1, y1, x2, y2 = minion.position
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            color = (255, 255, 0) if minion.can_attack else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        for minion in state.opponent_minions:
            x1, y1, x2, y2 = minion.position
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return overlay
