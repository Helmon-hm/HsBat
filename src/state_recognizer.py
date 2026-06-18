import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui
from PIL import Image

from src.logger import HsBatLogger
from src.paths import get_resource_path, get_templates_dir


@dataclass
class CardInfo:
    name: str
    cost: int
    position: Tuple[int, int, int, int]
    is_playable: bool = False


@dataclass
class MinionInfo:
    health: int
    attack: int
    position: Tuple[int, int, int, int]
    can_attack: bool = False


@dataclass
class GameState:
    is_our_turn: bool = False
    our_health: int = 0
    opponent_health: int = 0
    our_mana: int = 0
    total_mana: int = 0
    hand_cards: List[CardInfo] = field(default_factory=list)
    our_minions: List[MinionInfo] = field(default_factory=list)
    opponent_minions: List[MinionInfo] = field(default_factory=list)
    has_end_turn_button: bool = False
    is_game_over: bool = False
    screenshot: Optional[np.ndarray] = None
    timestamp: float = 0.0


class StateRecognizer:
    def __init__(self, config: dict):
        self.cfg = config
        self.logger = HsBatLogger().get_logger("StateRecognizer")
        self.scale = config["screen"]["scale"]
        self.ocr_lang = config["ocr"].get("lang", "eng")
        self.card_cost_offset = config["ocr"]["card_cost_offset"]

        self.tesseract_available = self._check_tesseract()
        self._load_templates()

    def _check_tesseract(self) -> bool:
        ocr_cfg = self.cfg["ocr"]
        tesseract_path = ocr_cfg.get("tesseract_path", "")

        if tesseract_path and os.path.exists(tesseract_path):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                pytesseract.get_tesseract_version()
                self.logger.info(f"Tesseract-OCR 已就绪: {tesseract_path}")
                return True
            except Exception as e:
                self.logger.warning(f"指定的 Tesseract 路径无法使用: {tesseract_path} ({e})")

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.logger.info("Tesseract-OCR 已就绪 (PATH)")
            return True
        except Exception:
            self.logger.warning(
                "Tesseract-OCR 未安装或不在 PATH 中。"
                "OCR 功能将不可用，部分识别会使用降级方案。\n"
                "安装方法: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "安装后配置 config.yaml 中的 ocr.tesseract_path"
            )
            return False

    def _safe_ocr(self, img: np.ndarray, config_str: str = "", lang: Optional[str] = None) -> str:
        if not self.tesseract_available:
            return ""
        try:
            import pytesseract
            return pytesseract.image_to_string(
                img,
                config=config_str,
                lang=lang or self.ocr_lang,
            )
        except Exception:
            return ""

    def _load_templates(self):
        templates_dir = get_templates_dir()
        self.templates = {}
        template_names = {
            "end_turn": self.cfg["templates"]["end_turn_button"],
            "attack": self.cfg["templates"]["attack_button"],
        }
        for key, filename in template_names.items():
            path = os.path.join(templates_dir, filename)
            if not os.path.exists(path):
                path = get_resource_path(os.path.join("templates", filename))
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[key] = img
                    self.logger.info(f"加载模板: {filename}")
                else:
                    self.logger.warning(f"无法加载模板: {filename}")
            else:
                self.logger.warning(f"模板文件不存在: {path}")

    def capture_screen(self) -> np.ndarray:
        region = self.cfg["screen"]["game_region"]
        screenshot = pyautogui.screenshot(
            region=(region["x"], region["y"], region["width"], region["height"])
        )
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return img

    def _match_template(
        self, img: np.ndarray, template_key: str
    ) -> Tuple[bool, Tuple[int, int, int, int]]:
        if template_key not in self.templates:
            return False, (0, 0, 0, 0)
        template = self.templates[template_key]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        threshold = self.cfg["templates"]["match_threshold"]
        if max_val >= threshold:
            h, w = template.shape[:2]
            return True, (max_loc[0], max_loc[1], max_loc[0] + w, max_loc[1] + h)
        return False, (0, 0, 0, 0)

    def _extract_mana(self, img: np.ndarray) -> Tuple[int, int]:
        try:
            region = self.cfg["game"]["mana_region"]
            x, y, w, h = region
            roi = img[y : y + h, x : x + w]

            lower = np.array(self.cfg["game"]["mana_color_lower"], dtype=np.uint8)
            upper = np.array(self.cfg["game"]["mana_color_upper"], dtype=np.uint8)
            mask = cv2.inRange(roi, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_crystals = 0
            total_crystals = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                total_crystals += 1
                filled_crystals += 1

            return filled_crystals, total_crystals
        except Exception as e:
            self.logger.error(f"法力识别失败: {e}")
            return 0, 0

    def _extract_health(self, img: np.ndarray, region_key: str) -> int:
        try:
            region = self.cfg["game"][region_key]
            x, y, w, h = region
            roi = img[y : y + h, x : x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            text = text.strip()
            if text:
                return int(text)
            return 0
        except Exception as e:
            self.logger.error(f"血量识别失败 ({region_key}): {e}")
            return 0

    def _detect_hand_cards(self, img: np.ndarray) -> List[CardInfo]:
        cards = []
        try:
            region = self.cfg["game"]["hand_card_region"]
            hx, hy, hw, hh = region
            roi = img[hy : hy + hh, hx : hx + hw]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            card_rects = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 50 and h > 100 and h < hh * 1.5:
                    card_rects.append((x + hx, y + hy, w, h))

            card_rects.sort(key=lambda r: r[0])
            merged = self._merge_overlapping(card_rects)

            for i, (cx, cy, cw, ch) in enumerate(merged):
                cost = self._extract_card_cost(img, cx, cy, cw, ch)
                name = self._extract_card_name(img, cx, cy, cw, ch)
                cards.append(CardInfo(name=name, cost=cost, position=(cx, cy, cx + cw, cy + ch)))

        except Exception as e:
            self.logger.error(f"手牌识别失败: {e}")

        return cards

    def _merge_overlapping(self, rects: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not rects:
            return []
        merged = []
        current = rects[0]
        for r in rects[1:]:
            if r[0] - (current[0] + current[2]) < 20:
                new_w = max(current[0] + current[2], r[0] + r[2]) - current[0]
                current = (current[0], current[1], new_w, max(current[3], r[3]))
            else:
                merged.append(current)
                current = r
        merged.append(current)
        return merged

    def _extract_card_cost(self, img: np.ndarray, cx: int, cy: int, cw: int, ch: int) -> int:
        if not self.tesseract_available:
            return 0
        try:
            offset = self.card_cost_offset
            cost_x = cx + offset["dx"]
            cost_y = cy + offset["dy"]
            cost_w = offset["width"]
            cost_h = offset["height"]
            cost_x = max(0, cost_x)
            cost_y = max(0, cost_y)

            if cost_x + cost_w > img.shape[1] or cost_y + cost_h > img.shape[0]:
                return 0

            roi = img[cost_y : cost_y + cost_h, cost_x : cost_x + cost_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            text = self._safe_ocr(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            text = text.strip()
            if text and text.isdigit():
                return int(text)
            return 0
        except Exception:
            return 0

    def _extract_card_name(self, img: np.ndarray, cx: int, cy: int, cw: int, ch: int) -> str:
        if not self.tesseract_available:
            return "unknown"
        try:
            name_x = cx - 20
            name_y = cy + int(ch * 0.75)
            name_w = cw + 40
            name_h = 25
            name_x = max(0, name_x)
            name_y = max(0, name_y)

            if name_x + name_w > img.shape[1] or name_y + name_h > img.shape[0]:
                return "unknown"

            roi = img[name_y : name_y + name_h, name_x : name_x + name_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 7", lang=self.ocr_lang)
            return text.strip() or "unknown"
        except Exception:
            return "unknown"

    def _detect_minions(self, img: np.ndarray) -> Tuple[List[MinionInfo], List[MinionInfo]]:
        our_minions = []
        opponent_minions = []
        try:
            h, w = img.shape[:2]
            our_zone = img[int(h * 0.6) : int(h * 0.8), int(w * 0.1) : int(w * 0.9)]
            opp_zone = img[int(h * 0.1) : int(h * 0.35), int(w * 0.1) : int(w * 0.9)]

            our_minions = self._find_minions_in_zone(our_zone, int(h * 0.6), int(w * 0.1))
            opponent_minions = self._find_minions_in_zone(opp_zone, int(h * 0.1), int(w * 0.1))

        except Exception as e:
            self.logger.error(f"随从识别失败: {e}")

        return our_minions, opponent_minions

    def _find_minions_in_zone(
        self, zone: np.ndarray, offset_y: int, offset_x: int
    ) -> List[MinionInfo]:
        minions = []
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 40 and w < 200 and h < 300:
                rects.append((x, y, w, h))

        rects.sort(key=lambda r: r[0])
        merged = self._merge_overlapping(rects)

        for mx, my, mw, mh in merged:
            abs_x = mx + offset_x
            abs_y = my + offset_y
            attack_roi = zone[my : my + 20, mx : mx + 30]
            health_roi = zone[my : my + 20, mx + mw - 30 : mx + mw]
            attack = self._ocr_number_from_roi(attack_roi)
            health = self._ocr_number_from_roi(health_roi)
            minions.append(
                MinionInfo(
                    health=health,
                    attack=attack,
                    position=(abs_x, abs_y, abs_x + mw, abs_y + mh),
                    can_attack=False,
                )
            )

        return minions

    def _ocr_number_from_roi(self, roi: np.ndarray) -> int:
        if not self.tesseract_available:
            return 0
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            text = self._safe_ocr(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            text = text.strip()
            if text and text.isdigit():
                return int(text)
            return 0
        except Exception:
            return 0

    def _detect_turn(self, img: np.ndarray) -> bool:
        if not self.tesseract_available:
            return self.detect_end_turn_button(img)
        try:
            region = self.cfg["game"]["turn_indicator_region"]
            x, y, w, h = region
            roi = img[y : y + h, x : x + w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 6")
            text = text.strip().lower()
            our_keywords = ["你的回合", "你", "your turn", "your"]
            for kw in our_keywords:
                if kw in text:
                    return True
            return False
        except Exception:
            return self.detect_end_turn_button(img)

    def detect_end_turn_button(self, img: np.ndarray) -> bool:
        found, _ = self._match_template(img, "end_turn")
        if found:
            return True
        if not self.tesseract_available:
            return False
        try:
            h, w = img.shape[:2]
            btn_roi = img[int(h * 0.85) : h, int(w * 0.8) : w]
            gray = cv2.cvtColor(btn_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 6")
            text = text.strip().lower()
            if "结束" in text or "回合" in text or "end" in text or "turn" in text:
                return True
            return False
        except Exception:
            return False

    def recognize(self) -> GameState:
        img = self.capture_screen()
        game_state = GameState(screenshot=img, timestamp=time.time())

        game_state.is_our_turn = self._detect_turn(img)
        game_state.our_health = self._extract_health(img, "health_region")
        game_state.opponent_health = self._extract_health(img, "enemy_health_region")
        game_state.our_mana, game_state.total_mana = self._extract_mana(img)
        game_state.hand_cards = self._detect_hand_cards(img)
        game_state.our_minions, game_state.opponent_minions = self._detect_minions(img)
        game_state.has_end_turn_button = self.detect_end_turn_button(img)

        mana = game_state.our_mana
        for card in game_state.hand_cards:
            card.is_playable = card.cost <= mana and card.cost > 0

        self.logger.debug(
            f"状态: 回合={'我方' if game_state.is_our_turn else '敌方'} "
            f"血量={game_state.our_health}/{game_state.opponent_health} "
            f"法力={game_state.our_mana}/{game_state.total_mana} "
            f"手牌={len(game_state.hand_cards)}张 "
            f"我方随从={len(game_state.our_minions)} 敌方随从={len(game_state.opponent_minions)}"
        )

        return game_state
