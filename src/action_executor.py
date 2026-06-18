import random
import time
from typing import List, Optional, Tuple

import numpy as np
import pyautogui

from src.logger import HsBatLogger


class ActionExecutor:
    def __init__(self, config: dict):
        self.cfg = config["action"]
        self.screen_cfg = config["screen"]
        self.logger = HsBatLogger().get_logger("ActionExecutor")
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

    def _random_delay(self, key: str) -> float:
        limits = self.cfg[key]
        return random.uniform(limits["min"], limits["max"])

    def _bezier_curve(
        self, start: Tuple[int, int], end: Tuple[int, int], num_points: int = 20
    ) -> List[Tuple[int, int]]:
        offset_range = self.cfg["bezier_control_offset"]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        offset_mag = random.uniform(offset_range["min"], offset_range["max"])
        offset_angle = random.uniform(0, 2 * np.pi)
        cp1 = (
            start[0] + dx * 0.25 + offset_mag * np.cos(offset_angle),
            start[1] + dy * 0.25 + offset_mag * np.sin(offset_angle),
        )
        cp2 = (
            start[0] + dx * 0.75 + offset_mag * np.cos(offset_angle + np.pi * 0.5),
            start[1] + dy * 0.75 + offset_mag * np.sin(offset_angle + np.pi * 0.5),
        )

        points = []
        for t in np.linspace(0, 1, num_points):
            x = (
                (1 - t) ** 3 * start[0]
                + 3 * (1 - t) ** 2 * t * cp1[0]
                + 3 * (1 - t) * t**2 * cp2[0]
                + t**3 * end[0]
            )
            y = (
                (1 - t) ** 3 * start[1]
                + 3 * (1 - t) ** 2 * t * cp1[1]
                + 3 * (1 - t) * t**2 * cp2[1]
                + t**3 * end[1]
            )
            points.append((int(x), int(y)))
        return points

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def _apply_offset(self, x: int, y: int) -> Tuple[int, int]:
        return (
            int(x * self.screen_cfg["scale"] + self.screen_cfg["game_region"]["x"]),
            int(y * self.screen_cfg["scale"] + self.screen_cfg["game_region"]["y"]),
        )

    def move_mouse(self, x: int, y: int):
        sx, sy = self._apply_offset(x, y)
        start_x, start_y = pyautogui.position()
        smooth = self.cfg.get("bezier_points", 20)
        path = self._bezier_curve((start_x, start_y), (sx, sy), smooth)
        for px, py in path:
            pyautogui.moveTo(px, py, duration=0.01)
            time.sleep(self._random_delay("mouse_move_delay") / len(path))
        pyautogui.moveTo(sx, sy, duration=0.01)

    def click(self, x: int, y: int):
        self.move_mouse(x, y)
        time.sleep(self._random_delay("rest_delay"))
        pyautogui.click()
        time.sleep(self._random_delay("click_delay"))

    def click_bbox(self, bbox: Tuple[int, int, int, int]):
        cx, cy = self._get_center(bbox)
        self.click(cx, cy)

    def drag(self, start: Tuple[int, int], end: Tuple[int, int]):
        sx, sy = self._apply_offset(*start)
        ex, ey = self._apply_offset(*end)
        pyautogui.moveTo(sx, sy, duration=0.05)
        time.sleep(self._random_delay("rest_delay"))
        pyautogui.mouseDown()
        time.sleep(self._random_delay("mouse_move_delay"))
        path = self._bezier_curve((sx, sy), (ex, ey))
        duration = self._random_delay("drag_duration")
        step_duration = duration / len(path)
        for px, py in path:
            pyautogui.moveTo(px, py, duration=step_duration)
        pyautogui.mouseUp()
        time.sleep(self._random_delay("click_delay"))

    def play_card(self, card_bbox: Tuple[int, int, int, int], target_bbox: Optional[Tuple[int, int, int, int]] = None):
        card_center = self._get_center(card_bbox)
        cx, cy = self._apply_offset(*card_center)
        pyautogui.moveTo(cx, cy, duration=0.05)
        time.sleep(self._random_delay("rest_delay"))
        pyautogui.mouseDown()
        time.sleep(0.05)

        if target_bbox:
            target_center = self._get_center(target_bbox)
            tx, ty = self._apply_offset(*target_center)
            path = self._bezier_curve((cx, cy), (tx, ty))
            duration = self._random_delay("drag_duration")
            step_duration = duration / len(path)
            for px, py in path:
                pyautogui.moveTo(px, py, duration=step_duration)
        else:
            screen_h = self.screen_cfg["game_region"]["height"]
            screen_w = self.screen_cfg["game_region"]["width"]
            play_y = cy - random.randint(int(0.093 * screen_h), int(0.185 * screen_h))
            play_x = cx + random.randint(-int(0.016 * screen_w), int(0.016 * screen_w))
            path = self._bezier_curve((cx, cy), (play_x, play_y))
            duration = self._random_delay("drag_duration")
            step_duration = duration / len(path)
            for px, py in path:
                pyautogui.moveTo(px, py, duration=step_duration)

        pyautogui.mouseUp()
        time.sleep(self._random_delay("click_delay"))

    def attack_with_minion(
        self, attacker_bbox: Tuple[int, int, int, int], target_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        self.click_bbox(attacker_bbox)
        time.sleep(self._random_delay("rest_delay"))
        if target_bbox:
            self.click_bbox(target_bbox)
        else:
            screen_w = self.screen_cfg["game_region"]["width"]
            screen_h = self.screen_cfg["game_region"]["height"]
            self.click(screen_w // 2, int(0.046 * screen_h))

    def end_turn(self, button_bbox: Optional[Tuple[int, int, int, int]] = None):
        screen_cfg = self.screen_cfg["game_region"]
        if button_bbox:
            self.click_bbox(button_bbox)
        else:
            self.click(
                int(screen_cfg["width"] * (1.0 - 0.052)),
                int(screen_cfg["height"] * (1.0 - 0.093)),
            )

    def click_screen_region(self, x: int, y: int):
        self.click(x, y)
        time.sleep(self._random_delay("click_delay"))

    def wait_for_turn(self, seconds: float = 45.0):
        self.logger.info(f"等待敌方回合结束，最长等待{seconds}秒")
        time.sleep(random.uniform(1.0, 2.0))
