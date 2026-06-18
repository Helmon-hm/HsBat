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
    is_post_game: bool = False
    is_main_menu: bool = False
    screenshot: Optional[np.ndarray] = None
    timestamp: float = 0.0


class StateRecognizer:
    def __init__(self, config: dict):
        self.cfg = config
        self.logger = HsBatLogger().get_logger("StateRecognizer")
        self._auto_detect_screen_size()
        self.scale = config["screen"]["scale"]
        self.ocr_lang = config["ocr"].get("lang", "eng")
        self.card_cost_offset = config["ocr"]["card_cost_offset"]

        self.tesseract_available = self._check_tesseract()
        self._load_templates()
        self._debug_dir = config["debug"].get("screenshot_dir", "screenshots")
        self._last_debug_save = 0.0

    def _auto_detect_screen_size(self):
        region = self.cfg["screen"]["game_region"]
        w = region.get("width", 0)
        h = region.get("height", 0)
        if w <= 0 or h <= 0:
            screen_w, screen_h = pyautogui.size()
            region["width"] = screen_w
            region["height"] = screen_h
            self.logger.info(f"自动检测屏幕分辨率: {screen_w} x {screen_h}")

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

    def _fractional_to_pixels(self, region: list, img_w: int, img_h: int) -> list:
        if not region or len(region) < 2:
            return region
        if all(0 <= v <= 1 for v in region):
            converted = []
            for i, v in enumerate(region):
                if i % 2 == 0:
                    converted.append(int(v * img_w))
                else:
                    converted.append(int(v * img_h))
            return converted
        return region

    def _game_region_pixels(self, key: str, img: np.ndarray) -> list:
        region = self.cfg["game"].get(key, [0, 0, 0, 0])
        h, w = img.shape[:2]
        return self._fractional_to_pixels(region, w, h)

    def _extract_mana(self, img: np.ndarray) -> Tuple[int, int]:
        try:
            region = self._game_region_pixels("mana_region", img)
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
                min_area = int(0.000014 * img.shape[0] * img.shape[1])
                if area < min_area:
                    continue
                total_crystals += 1
                filled_crystals += 1

            return filled_crystals, total_crystals
        except Exception as e:
            self.logger.error(f"法力识别失败: {e}")
            return 0, 0

    def _extract_health(self, img: np.ndarray, region_key: str) -> int:
        try:
            region = self._game_region_pixels(region_key, img)
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
            region = self._game_region_pixels("hand_card_region", img)
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
                min_card_w = int(0.026 * img.shape[1])
                min_card_h = int(0.093 * img.shape[0])
                if w > min_card_w and h > min_card_h and h < hh * 1.5:
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

    def _merge_overlapping(self, rects: List[Tuple[int, int, int, int]], gap: int = 20) -> List[Tuple[int, int, int, int]]:
        if not rects:
            return []
        merged = []
        current = rects[0]
        for r in rects[1:]:
            if r[0] - (current[0] + current[2]) < gap:
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
            img_h, img_w = img.shape[:2]
            name_offset_x = int(0.01 * img_w)
            name_offset_w = int(0.021 * img_w)
            name_h = int(0.023 * img_h)
            name_x = cx - name_offset_x
            name_y = cy + int(ch * 0.75)
            name_w = cw + name_offset_w
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

            our_minions = self._find_minions_in_zone(our_zone, int(h * 0.6), int(w * 0.1), img)
            opponent_minions = self._find_minions_in_zone(opp_zone, int(h * 0.1), int(w * 0.1), img)

        except Exception as e:
            self.logger.error(f"随从识别失败: {e}")

        return our_minions, opponent_minions

    def _find_minions_in_zone(
        self, zone: np.ndarray, offset_y: int, offset_x: int, full_img: np.ndarray
    ) -> List[MinionInfo]:
        minions = []
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = full_img.shape[:2]
        min_w = int(self.cfg["game"].get("minion_min_size", [0.016, 0.037])[0] * img_w)
        min_h = int(self.cfg["game"].get("minion_min_size", [0.016, 0.037])[1] * img_h)
        max_w = int(self.cfg["game"].get("minion_max_size", [0.104, 0.278])[0] * img_w)
        max_h = int(self.cfg["game"].get("minion_max_size", [0.104, 0.278])[1] * img_h)
        ocr_w = int(self.cfg["game"].get("minion_ocr_size", [0.016, 0.019])[0] * img_w)
        ocr_h = int(self.cfg["game"].get("minion_ocr_size", [0.016, 0.019])[1] * img_h)
        merge_gap = int(0.01 * img_w)

        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_w and h > min_h and w < max_w and h < max_h:
                rects.append((x, y, w, h))

        rects.sort(key=lambda r: r[0])
        merged = self._merge_overlapping(rects, merge_gap)

        for mx, my, mw, mh in merged:
            abs_x = mx + offset_x
            abs_y = my + offset_y
            attack_roi = zone[my : my + ocr_h, mx : mx + ocr_w]
            health_roi = zone[my : my + ocr_h, mx + mw - ocr_w : mx + mw]
            attack = self._ocr_number_from_roi(attack_roi)
            health = self._ocr_number_from_roi(health_roi)
            can_attack = self._detect_attackable_green_border(
                full_img, abs_x, abs_y, mw, mh
            )
            minions.append(
                MinionInfo(
                    health=health,
                    attack=attack,
                    position=(abs_x, abs_y, abs_x + mw, abs_y + mh),
                    can_attack=can_attack,
                )
            )

        return minions

    def _detect_attackable_green_border(
        self, img: np.ndarray, minion_x: int, minion_y: int, minion_w: int, minion_h: int
    ) -> bool:
        try:
            border_w = self.cfg["game"].get("attackable_green_border_width", 8)
            green_lower = np.array(self.cfg["game"]["attackable_green_lower"], dtype=np.uint8)
            green_upper = np.array(self.cfg["game"]["attackable_green_upper"], dtype=np.uint8)
            min_ratio = self.cfg["game"].get("attackable_green_min_ratio", 0.08)

            bx1 = max(0, minion_x - border_w)
            by1 = max(0, minion_y - border_w)
            bx2 = min(img.shape[1], minion_x + minion_w + border_w)
            by2 = min(img.shape[0], minion_y + minion_h + border_w)

            if bx2 <= bx1 or by2 <= by1:
                return False

            border_region = img[by1:by2, bx1:bx2]
            hsv = cv2.cvtColor(border_region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, green_lower, green_upper)
            total_pixels = mask.size
            if total_pixels == 0:
                return False
            green_pixels = cv2.countNonZero(mask)
            ratio = green_pixels / total_pixels
            return ratio >= min_ratio
        except Exception:
            return False

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
            region = self._game_region_pixels("turn_indicator_region", img)
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

    def detect_game_over(self, img: np.ndarray) -> bool:
        our_health = self._extract_health(img, "health_region")
        opp_health = self._extract_health(img, "enemy_health_region")
        has_btn = self.detect_end_turn_button(img)

        if our_health == 0 and opp_health == 0 and not has_btn:
            return True

        if not self.tesseract_available:
            return False

        try:
            h, w = img.shape[:2]
            center_roi = img[int(h * 0.35):int(h * 0.65), int(w * 0.3):int(w * 0.7)]
            gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 6")
            text_lower = text.strip().lower()
            for kw in ["胜利", "失败", "victory", "defeat", "you win", "you lose",
                        "经验", "exp"]:
                if kw in text_lower:
                    return True
        except Exception:
            pass
        return False

    def detect_main_menu(self, img: np.ndarray) -> bool:
        region = self.cfg["game"].get("play_button_region", [0.36, 0.56, 0.27, 0.06])
        if not region or len(region) < 4:
            return False
        region = self._game_region_pixels("play_button_region", img)
        x, y, w_box, h_box = region[0], region[1], region[2], region[3]
        if not self.tesseract_available:
            return False
        try:
            roi = img[y:y + h_box, x:x + w_box]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = self._safe_ocr(thresh, config="--psm 6")
            text_lower = text.strip().lower()
            for kw in ["play", "开始", "对战", "排位", "休闲", "hplay", "bplay",
                        "standard", "wild", "casual", "ranked"]:
                if kw in text_lower:
                    return True
        except Exception:
            pass
        return False

    def detect_post_game(self, img: np.ndarray) -> bool:
        if self.detect_game_over(img):
            return True
        our_health = self._extract_health(img, "health_region")
        opp_health = self._extract_health(img, "enemy_health_region")
        has_btn = self.detect_end_turn_button(img)
        if our_health == 0 and opp_health == 0 and not has_btn:
            return True
        return False

    def _save_debug_crops(self, img: np.ndarray, game_state: GameState):
        if not self.cfg["debug"].get("save_screenshots", False):
            return
        now = time.time()
        if now - self._last_debug_save < 1.5:
            return
        self._last_debug_save = now

        out_dir = os.path.join(self._debug_dir, "latest")
        os.makedirs(out_dir, exist_ok=True)

        def _save(name, roi):
            path = os.path.join(out_dir, f"{name}.png")
            try:
                cv2.imwrite(path, roi)
            except Exception:
                pass

        _save("00_full", img)
        full_path = os.path.join(self._debug_dir, "latest_full.png")
        try:
            cv2.imwrite(full_path, img)
        except Exception:
            pass

        keys = ["mana_region", "health_region", "enemy_health_region",
                "hand_card_region", "turn_indicator_region"]
        for key in keys:
            region = self._game_region_pixels(key, img)
            x, y, w, h = region
            if w > 0 and h > 0:
                roi = img[y:y + h, x:x + w]
                _save(key, roi)

        for i, card in enumerate(game_state.hand_cards):
            x1, y1, x2, y2 = card.position
            if x2 > x1 and y2 > y1:
                _save(f"card_{i:02d}_{card.name}", img[y1:y2, x1:x2])

        for i, m in enumerate(game_state.our_minions):
            x1, y1, x2, y2 = m.position
            if x2 > x1 and y2 > y1:
                tag = "atk" if m.can_attack else "idle"
                _save(f"our_minion_{i:02d}_{tag}", img[y1:y2, x1:x2])

        for i, m in enumerate(game_state.opponent_minions):
            x1, y1, x2, y2 = m.position
            if x2 > x1 and y2 > y1:
                _save(f"opp_minion_{i:02d}", img[y1:y2, x1:x2])

        self.logger.debug(f"调试截图已保存: {out_dir}")

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
        game_state.is_game_over = self.detect_game_over(img)
        game_state.is_post_game = game_state.is_game_over
        game_state.is_main_menu = self.detect_main_menu(img)

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

        self._save_debug_crops(img, game_state)

        return game_state
