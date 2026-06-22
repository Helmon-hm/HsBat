import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import pyautogui

from src.logger import HsBatLogger
from src.paths import get_resource_path, get_templates_dir
from src.log_tracker import HearthstoneLogTracker


@dataclass
class CardInfo:
    name: str
    cost: int
    position: Tuple[int, int, int, int]
    is_playable: bool = False
    card_type: str = "unknown"
    card_id: str = ""
    zone_position: int = 0


@dataclass
class MinionInfo:
    health: int
    attack: int
    position: Tuple[int, int, int, int]
    can_attack: bool = False
    name: str = ""
    card_id: str = ""
    has_taunt: bool = False
    has_divine_shield: bool = False
    has_stealth: bool = False
    is_elusive: bool = False
    has_poisonous: bool = False
    has_reborn: bool = False
    has_deathrattle: bool = False
    has_battlecry: bool = False
    has_lifesteal: bool = False
    has_rush: bool = False
    has_charge: bool = False
    is_frozen: bool = False
    is_silenced: bool = False
    card_race: str = ""


@dataclass
class GameState:
    is_our_turn: bool = False
    our_health: int = 0
    opponent_health: int = 0
    our_mana: int = 0
    total_mana: int = 0
    our_armor: int = 0
    opponent_armor: int = 0
    hand_cards: List[CardInfo] = field(default_factory=list)
    our_minions: List[MinionInfo] = field(default_factory=list)
    opponent_minions: List[MinionInfo] = field(default_factory=list)
    has_end_turn_button: bool = False
    is_end_turn_green: bool = False
    end_turn_button_bbox: Optional[Tuple[int, int, int, int]] = None
    is_game_over: bool = False
    is_post_game: bool = False
    is_main_menu: bool = False
    screenshot: Optional[np.ndarray] = None
    timestamp: float = 0.0
    log_data_available: bool = False
    hero_power_used: bool = False
    # From DebugPrintOptions: available actions this turn
    action_options: List = field(default_factory=list)


class StateRecognizer:
    def __init__(self, config: dict):
        self.cfg = config
        self.logger = HsBatLogger().get_logger("StateRecognizer")
        self._auto_detect_screen_size()
        self.scale = config["screen"]["scale"]
        self.ocr_lang = config["ocr"].get("lang", "eng")
        self.ocr_upscale = config["ocr"].get("upscale_factor", 3)
        self.card_cost_offset = config["ocr"]["card_cost_offset"]

        self.recog_cfg = config.get("recognition", {})
        self.tesseract_available = self._check_tesseract()
        self.paddle_ocr = self._init_paddleocr()
        self._load_templates()
        self._debug_dir = config["debug"].get("screenshot_dir", "screenshots")
        self._last_debug_save = 0.0

        cache_size = self.recog_cfg.get("frame_cache_size", 5)
        self._health_cache: Deque[int] = deque(maxlen=cache_size)
        self._enemy_health_cache: Deque[int] = deque(maxlen=cache_size)
        self._mana_cache: Deque[int] = deque(maxlen=cache_size)
        self._turn_cache: Deque[bool] = deque(maxlen=cache_size)
        self._consistent_needed = self.recog_cfg.get("min_consistent_frames", 3)

        self._log_tracker = HearthstoneLogTracker(
            manual_path=config.get("log_tracking", {}).get("power_log_path", None)
        )
        self._log_enabled = config.get("log_tracking", {}).get("enabled", True)
        self._last_log_warning = 0.0

        self._digit_color_masks = [
            ("white", np.array([150, 150, 150], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8)),
            ("red", np.array([0, 0, 140], dtype=np.uint8), np.array([80, 80, 255], dtype=np.uint8)),
            ("green", np.array([0, 140, 0], dtype=np.uint8), np.array([100, 255, 100], dtype=np.uint8)),
            ("bright_white", np.array([180, 180, 180], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8)),
        ]

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

    def _init_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang='en', use_angle_cls=False)
            self.logger.info("PaddleOCR 已就绪")
            return ocr
        except Exception as e:
            self.logger.warning(f"PaddleOCR 初始化失败: {e}，将使用 Tesseract 回退")
            return None

    def _ocr_number(self, roi: np.ndarray, whitelist: str = "0123456789") -> int:
        """Unified OCR: PaddleOCR first, Tesseract fallback."""
        if roi is None or roi.size == 0:
            return 0

        # PaddleOCR
        if self.paddle_ocr is not None:
            try:
                if len(roi.shape) == 2:
                    ocr_input = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                else:
                    ocr_input = roi
                # Upscale small images for better text detection
                h, w = ocr_input.shape[:2]
                if w < 100 or h < 100:
                    scale = max(3, min(6, 100 // min(w, h)))
                    ocr_input = cv2.resize(ocr_input, None,
                                           fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC)
                result = self.paddle_ocr.ocr(ocr_input)
                if result and result[0]:
                    all_digits = []
                    for line in result[0]:
                        text = line[1][0].strip()
                        digits = ''.join(c for c in text if c in whitelist)
                        if digits:
                            all_digits.append((line[0][0][0], digits))
                    if all_digits:
                        all_digits.sort(key=lambda x: x[0])
                        combined = ''.join(d for _, d in all_digits)
                        if combined.isdigit():
                            val = int(combined)
                            if 0 <= val <= 20:
                                return val
                    elif all_digits:
                        val = int(all_digits[0][1])
                        if 0 <= val <= 20:
                            return val
            except Exception:
                pass

        # Tesseract fallback
        return self._tesseract_ocr_number(roi)

    def _tesseract_ocr_number(self, roi: np.ndarray) -> int:
        import pytesseract
        if not self.tesseract_available or roi is None or roi.size == 0:
            return 0
        try:
            if len(roi.shape) == 2:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            else:
                roi_bgr = roi
            if self.ocr_upscale > 1:
                roi_bgr = cv2.resize(roi_bgr, None, fx=self.ocr_upscale, fy=self.ocr_upscale,
                                     interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            for method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
                _, thresh = cv2.threshold(gray, 0, 255, method)
                if cv2.countNonZero(thresh) < 8:
                    continue
                for psm in [7, 8, 6]:
                    text = pytesseract.image_to_string(
                        thresh, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                    ).strip()
                    if text and text.isdigit():
                        return int(text)
            for label, lower, upper in self._digit_color_masks:
                mask = cv2.inRange(roi_bgr, lower, upper)
                if cv2.countNonZero(mask) < 10:
                    continue
                text = pytesseract.image_to_string(
                    mask, config="--psm 7 -c tessedit_char_whitelist=0123456789"
                ).strip()
                if text and text.isdigit():
                    return int(text)
        except Exception:
            pass
        return 0

    def _debug_save_roi(self, name: str, roi: np.ndarray):
        if not self.cfg["debug"].get("save_screenshots", False):
            return
        out_dir = os.path.join(self._debug_dir, "ocr_debug")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.png")
        try:
            cv2.imwrite(path, roi)
        except Exception:
            pass

    def _preprocess_for_ocr(self, img: np.ndarray, invert: bool = False) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        upscale = self.recog_cfg.get("ocr_upscale", self.cfg["ocr"].get("upscale_factor", 3))
        if upscale > 1:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def _safe_ocr(self, img: np.ndarray, config: str = "", lang: Optional[str] = None) -> str:
        if not self.tesseract_available:
            return ""
        try:
            import pytesseract
            processed = self._preprocess_for_ocr(img)
            return pytesseract.image_to_string(processed, config=config, lang=lang or self.ocr_lang)
        except Exception:
            return ""

    def _safe_ocr_number(self, roi: np.ndarray) -> int:
        if not self.tesseract_available:
            return 0
        if roi is None or roi.size == 0:
            return 0
        try:
            import pytesseract

            upscale = self.ocr_upscale

            if len(roi.shape) == 2:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            else:
                roi_bgr = roi

            if upscale > 1:
                roi_bgr = cv2.resize(roi_bgr, None, fx=upscale, fy=upscale,
                                     interpolation=cv2.INTER_CUBIC)

            best_result = 0

            # OTSU: works for any color with contrast
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            for method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
                _, thresh = cv2.threshold(gray, 0, 255, method)
                if cv2.countNonZero(thresh) < 8:
                    continue
                for psm in [7, 8, 6]:
                    text = pytesseract.image_to_string(
                        thresh, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                    ).strip()
                    if text and text.isdigit():
                        val = int(text)
                        if val > best_result:
                            best_result = val

            # Color masks: specific for white/red/green digits
            for label, lower, upper in self._digit_color_masks:
                mask = cv2.inRange(roi_bgr, lower, upper)
                if cv2.countNonZero(mask) < 10:
                    continue
                for psm in [7, 8, 6]:
                    text = pytesseract.image_to_string(
                        mask, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                    ).strip()
                    if text and text.isdigit():
                        val = int(text)
                        if val > best_result:
                            best_result = val

            if best_result > 0:
                return best_result

            self._debug_save_roi("number_fail_original", roi)
            return 0

        except Exception:
            return 0

    def _load_templates(self):
        templates_dir = get_templates_dir()
        self.templates = {}
        template_names = {
            "end_turn": self.cfg["templates"]["end_turn_button"],
            "play": self.cfg["templates"].get("play_button", "play_button.png"),
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
                if key == "end_turn":
                    self.logger.warning(f"模板文件不存在: {path}")

        turn_extra = {
            "end_turn_yellow": "end_turn.png",
            "end_turn_green": "end_turn2.png",
            "end_turn_opponent": "end_turn3.png",
        }
        for key, filename in turn_extra.items():
            path = os.path.join(templates_dir, filename)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[key] = img
                    self.logger.info(f"加载回合模板: {filename}")
        if any(k in self.templates for k in turn_extra):
            self.logger.info(f"回合模板已就绪: {[k for k in turn_extra if k in self.templates]}")

        self._cost_templates = {}
        cost_dir = os.path.join(templates_dir, "digits")
        if os.path.isdir(cost_dir):
            for digit in range(0, 21):
                path = os.path.join(cost_dir, f"{digit}.png")
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self._cost_templates[digit] = img
            if self._cost_templates:
                self.logger.info(f"加载费用模板: {sorted(self._cost_templates.keys())}")
            else:
                self.logger.warning("费用模板目录为空: templates/digits/")
        else:
            self.logger.info("费用模板目录不存在,使用OCR降级方案")

    def capture_screen(self) -> np.ndarray:
        region = self.cfg["screen"]["game_region"]
        screenshot = pyautogui.screenshot(
            region=(region["x"], region["y"], region["width"], region["height"])
        )
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return img

    def _match_template(self, img: np.ndarray, template_key: str) -> Tuple[bool, Tuple[int, int, int, int]]:
        if template_key not in self.templates:
            return False, (0, 0, 0, 0)
        template = self.templates[template_key]
        threshold = self.cfg["templates"]["match_threshold"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        best_val = 0
        best_bbox = (0, 0, 0, 0)
        for sc in [0.8, 0.9, 1.0, 1.1, 1.2]:
            sw = int(template.shape[1] * sc)
            sh = int(template.shape[0] * sc)
            if sw < 10 or sh < 10 or sw > gray.shape[1] or sh > gray.shape[0]:
                continue
            st = cv2.resize(template, (sw, sh))
            result = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_bbox = (max_loc[0], max_loc[1], max_loc[0] + sw, max_loc[1] + sh)
        if best_val >= threshold:
            return True, best_bbox
        return False, (0, 0, 0, 0)

    def _match_any_template(self, img: np.ndarray, template_keys: List[str]) -> Tuple[bool, str, Tuple[int, int, int, int]]:
        best_val = 0
        best_key = ""
        best_bbox = (0, 0, 0, 0)
        threshold = self.cfg["templates"]["match_threshold"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        for key in template_keys:
            if key not in self.templates:
                continue
            template = self.templates[key]
            for sc in [0.8, 0.9, 1.0, 1.1, 1.2]:
                sw = int(template.shape[1] * sc)
                sh = int(template.shape[0] * sc)
                if sw < 10 or sh < 10 or sw > gray.shape[1] or sh > gray.shape[0]:
                    continue
                st = cv2.resize(template, (sw, sh))
                result = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val >= threshold and max_val > best_val:
                    best_val = max_val
                    best_key = key
                    best_bbox = (max_loc[0], max_loc[1], max_loc[0] + sw, max_loc[1] + sh)
        if best_key:
            return True, best_key, best_bbox
        return False, "", (0, 0, 0, 0)

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

    def _crop_region(self, key: str, img: np.ndarray):
        x, y, w, h = self._game_region_pixels(key, img)
        if w <= 0 or h <= 0:
            return None
        return img[y:y + h, x:x + w]

    def _crop_all_regions(self, img: np.ndarray) -> dict:
        region_keys = ["mana_region", "mana_text_region", "health_region", "enemy_health_region",
                       "hand_card_region", "our_board_region", "enemy_board_region",
                       "turn_indicator_region", "end_turn_button_region", "play_button_region"]
        crops = {}
        for key in region_keys:
            crops[key] = self._crop_region(key, img)
        return crops

    def _extract_mana(self, crops: dict, img_hw: tuple) -> Tuple[int, int]:
        for key in ["mana_text_region", "mana_region"]:
            roi = crops.get(key)
            if roi is not None:
                result = self._ocr_mana_numbers(roi)
                if result != (0, 0):
                    return result
        return 0, 0

    def _ocr_mana_numbers(self, roi: np.ndarray) -> Tuple[int, int]:
        import re
        if roi is None or roi.size == 0:
            return 0, 0

        # PaddleOCR
        if self.paddle_ocr is not None:
            try:
                ocr_input = roi
                h, w = ocr_input.shape[:2]
                if w < 100 or h < 100:
                    scale = max(3, min(6, 100 // min(w, h)))
                    ocr_input = cv2.resize(ocr_input, None,
                                           fx=scale, fy=scale,
                                           interpolation=cv2.INTER_CUBIC)
                result = self.paddle_ocr.ocr(ocr_input)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0].strip()
                        m = re.search(r'(\d+)\s*/\s*(\d+)', text)
                        if m:
                            a, b = int(m.group(1)), int(m.group(2))
                            if 0 <= a <= 20 and 0 <= b <= 20:
                                return a, b
                        digits = re.findall(r'\d+', text)
                        if len(digits) >= 2:
                            a, b = int(digits[0]), int(digits[1])
                            if 0 <= a <= 20 and 0 <= b <= 20:
                                return a, b
            except Exception:
                pass

        return 0, 0


    def _extract_health(self, roi: np.ndarray, region_key: str) -> int:
        try:
            if roi is None or roi.size == 0:
                return 0
            return self._ocr_number(roi)
        except Exception as e:
            self.logger.error(f"血量识别失败 ({region_key}): {e}")
            return 0

    def _detect_hand_cards(self, roi: np.ndarray, img_hw: tuple, hand_offset: tuple) -> List[CardInfo]:
        cards = []
        try:
            if roi is None or roi.size == 0:
                return cards
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hh, hw = roi.shape[:2]
            hx, hy = hand_offset

            crystal_count, crystal_xs = self._detect_mana_crystals(roi)
            if crystal_count >= 1 and len(crystal_xs) == crystal_count:
                self.logger.debug(f"水晶检测: {crystal_count}张卡, 位置={crystal_xs}")
                boundaries = self._segment_by_crystals(hw, crystal_xs)
            else:
                card_count = self._estimate_card_count(roi_gray, img_hw[0], img_hw[1])
                if card_count < 1:
                    return cards
                self.logger.debug(f"边缘检测: {card_count}张卡 (水晶检测失败, 回退)")
                boundaries = self._get_card_boundaries(roi, card_count)

            for seg_x, seg_w in boundaries:
                if seg_w < 20:
                    continue
                seg = roi[:, max(0, seg_x): min(hw, seg_x + seg_w)]
                if seg.size == 0:
                    continue
                card_screen_x1 = hx + seg_x
                card_screen_x2 = hx + seg_x + seg_w
                card_screen_y1 = hy
                card_screen_y2 = hy + hh

                cost = self._ocr_card_cost_improved(seg)
                card_type = self._detect_card_type(seg)
                cards.append(CardInfo(
                    name="", cost=cost,
                    position=(card_screen_x1, card_screen_y1, card_screen_x2, card_screen_y2),
                    card_type=card_type,
                ))

        except Exception as e:
            self.logger.error(f"手牌识别失败: {e}")

        return cards

    def _ocr_small_digit(self, roi: np.ndarray) -> int:
        try:
            if roi is None or roi.size < 30:
                return -1

            # Template matching first (more reliable than OCR for small digits)
            if self._cost_templates:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
                best_val = 0.0
                best_digit = 0
                angles = self._get_rotate_angles()
                for digit, tmpl in self._cost_templates.items():
                    for sc in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
                        for angle in (angles if best_val < 0.60 else [0]):
                            if angle != 0.0:
                                rotated = self._rotate_template(tmpl, angle)
                                sw = int(rotated.shape[1] * sc)
                                sh = int(rotated.shape[0] * sc)
                                if sw < 4 or sh < 4 or sw > gray.shape[1] or sh > gray.shape[0]:
                                    continue
                                st = cv2.resize(rotated, (sw, sh))
                            else:
                                sw = int(tmpl.shape[1] * sc)
                                sh = int(tmpl.shape[0] * sc)
                                if sw < 4 or sh < 4 or sw > gray.shape[1] or sh > gray.shape[0]:
                                    continue
                                st = cv2.resize(tmpl, (sw, sh))
                            corr = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(corr)
                            if max_val > best_val:
                                best_val = max_val
                                best_digit = digit
                if best_val >= 0.45:
                    return best_digit

            # OCR fallback
            return self._ocr_number(roi)
        except Exception:
            return -1

    def _get_card_boundaries(self, roi: np.ndarray, card_count: int) -> List[Tuple[int, int]]:
        hh, hw = roi.shape[:2]
        card_width = hw / card_count
        return [(int(i * card_width), int(card_width)) for i in range(card_count)]

    def _detect_mana_crystals(self, hand_roi: np.ndarray) -> Tuple[int, List[int]]:
        try:
            hh, hw = hand_roi.shape[:2]
            top_h = max(20, int(hh * 0.30))
            top = hand_roi[:top_h, :]
            gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if len(top.shape) == 3 else top
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            lap_abs = np.abs(lap)
            col_sums = np.sum(lap_abs, axis=0)
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks
            sigma = max(2.0, hw / 80.0)
            col_smooth = gaussian_filter1d(col_sums.astype(float), sigma=sigma)
            threshold = np.max(col_smooth) * 0.20
            min_dist = max(15, hw // 18)
            peaks, props = find_peaks(col_smooth, distance=min_dist,
                                       height=threshold, prominence=threshold * 0.5)
            if len(peaks) == 0:
                threshold_low = np.max(col_smooth) * 0.08
                peaks, _ = find_peaks(col_smooth, distance=min_dist,
                                       height=threshold_low)
            peak_xs = [int(p) for p in peaks]
            self._debug_save_crystal_plot(hand_roi, col_smooth, peaks)
            return len(peak_xs), peak_xs
        except ImportError:
            self.logger.debug("scipy未安装，水晶检测不可用")
            return 0, []
        except Exception as e:
            self.logger.debug(f"水晶检测失败: {e}")
            return 0, []

    def _segment_by_crystals(self, roi_width: int, crystal_xs: List[int]) -> List[Tuple[int, int]]:
        if len(crystal_xs) < 1:
            return []
        if len(crystal_xs) == 1:
            cx = crystal_xs[0]
            left = max(0, cx - roi_width // 8)
            right = min(roi_width, cx + roi_width // 4)
            return [(left, right - left)]
        boundaries = []
        for i, cx in enumerate(crystal_xs):
            if i == 0:
                mid = (crystal_xs[0] + crystal_xs[1]) // 2
                left = max(0, cx - roi_width // 12)
                right = mid
            elif i == len(crystal_xs) - 1:
                mid = (crystal_xs[i - 1] + crystal_xs[i]) // 2
                left = mid
                right = min(roi_width, cx + roi_width // 8)
            else:
                mid_left = (crystal_xs[i - 1] + crystal_xs[i]) // 2
                mid_right = (crystal_xs[i] + crystal_xs[i + 1]) // 2
                left = mid_left
                right = mid_right
            w = max(30, right - left)
            boundaries.append((left, w))
        return boundaries

    def _extract_crystal_by_x(self, hand_roi: np.ndarray, cx: int) -> Optional[np.ndarray]:
        try:
            hh, hw = hand_roi.shape[:2]
            crystal_w = max(20, hw // 30)
            crystal_h = max(20, hh // 6)
            x1 = max(0, cx - crystal_w // 2)
            x2 = min(hw, cx + crystal_w // 2)
            y1 = 0
            y2 = min(hh, crystal_h)
            if x2 <= x1 or y2 <= y1:
                return None
            return hand_roi[y1:y2, x1:x2]
        except Exception:
            return None

    def _debug_save_crystal_plot(self, hand_roi: np.ndarray, col_smooth: np.ndarray, peaks: np.ndarray):
        if not self.cfg["debug"].get("save_screenshots", False):
            return
        try:
            out_dir = os.path.join(self._debug_dir, "ocr_debug")
            os.makedirs(out_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(out_dir, f"crystal_{ts}.png")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
            ax1.imshow(cv2.cvtColor(hand_roi[:max(20, int(hand_roi.shape[0]*0.3)), :],
                                     cv2.COLOR_BGR2RGB))
            for p in peaks:
                ax1.axvline(x=p, color='r', linewidth=1)
            ax1.set_title("Crystal Detection")
            ax2.plot(col_smooth)
            ax2.set_title("Laplacian Column Sums (Smoothed)")
            plt.tight_layout()
            plt.savefig(path, dpi=80)
            plt.close()
        except Exception:
            pass

    def _detect_card_type(self, card_segment: np.ndarray) -> str:
        try:
            h, w = card_segment.shape[:2]
            if h < 40 or w < 30:
                return "unknown"
            y_start = int(h * 0.72)
            bottom = card_segment[y_start:, :]
            if bottom.size == 0:
                return "unknown"
            bh = bottom.shape[0]
            mid_x = bottom.shape[1] // 2
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
            left_hsv = hsv[:, :mid_x, :]
            yellow_lower = np.array([15, 60, 60], dtype=np.uint8)
            yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
            yellow_mask = cv2.inRange(left_hsv, yellow_lower, yellow_upper)
            yellow_ratio = np.count_nonzero(yellow_mask) / yellow_mask.size
            right_hsv = hsv[:, mid_x:, :]
            red_lower1 = np.array([0, 80, 80], dtype=np.uint8)
            red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
            red_lower2 = np.array([160, 80, 80], dtype=np.uint8)
            red_upper2 = np.array([179, 255, 255], dtype=np.uint8)
            red_mask1 = cv2.inRange(right_hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(right_hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_ratio = np.count_nonzero(red_mask) / red_mask.size
            threshold = 0.015
            if yellow_ratio > threshold and red_ratio > threshold:
                return "minion"
            return "spell"
        except Exception:
            return "unknown"

    def _estimate_card_count(self, roi_gray: np.ndarray, img_h: int, img_w: int) -> int:
        method = self.recog_cfg.get("card_count_method", "sobel")
        if method == "sobel":
            return self._estimate_card_count_sobel(roi_gray)
        return self._estimate_card_count_canny(roi_gray, img_h, img_w)

    def _estimate_card_count_sobel(self, roi_gray: np.ndarray) -> int:
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks

            h, w = roi_gray.shape[:2]
            top_ratio = self.recog_cfg.get("sobel_top_ratio", 0.35)
            top_h = max(30, int(h * top_ratio))
            top = roi_gray[:top_h, :]

            sobel_x = cv2.Sobel(top, cv2.CV_64F, 1, 0, ksize=3)
            sobel_x_abs = np.abs(sobel_x)
            col_sums = np.sum(sobel_x_abs, axis=0)

            sigma = self.recog_cfg.get("sobel_sigma", 5)
            col_smooth = gaussian_filter1d(col_sums.astype(float), sigma=sigma)

            height_threshold = np.max(col_smooth) * 0.15
            prominence = self.recog_cfg.get("sobel_prominence", 0.08)
            prom_threshold = np.max(col_smooth) * prominence

            # Multi-scale: large distance (few cards) to small (many cards)
            # 3 scales: 2 should agree for any hand size, median picks winner
            distances = [max(40, int(w * 0.20)),
                         max(40, int(w * 0.12)),
                         max(40, int(w * 0.07))]
            counts = []
            for dist in distances:
                peaks, _ = find_peaks(col_smooth,
                                      distance=dist,
                                      height=height_threshold,
                                      prominence=prom_threshold)
                counts.append(len(peaks))

            # Median is robust: if 2/3 scales agree, median picks the consensus
            from statistics import median
            count = int(median(counts))
            if count < 1:
                self.logger.warning(f"Sobel卡牌计数为0，回退到Canny")
                return self._estimate_card_count_canny(roi_gray, roi_gray.shape[0], roi_gray.shape[0])
            return count
        except ImportError:
            self.logger.warning("scipy 未安装，回退到 Canny 边缘检测卡牌计数")
            return self._estimate_card_count_canny(roi_gray, roi_gray.shape[0], roi_gray.shape[0])

    def _estimate_card_count_canny(self, roi_gray: np.ndarray, img_h: int, img_w: int) -> int:
        cl = self.recog_cfg.get("canny_lower", 30)
        cu = self.recog_cfg.get("canny_upper", 100)
        edges = cv2.Canny(roi_gray, cl, cu)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_card_w = int(self.recog_cfg.get("card_min_width_ratio", 0.026) * img_w)
        min_card_h = int(self.recog_cfg.get("card_min_height_ratio", 0.093) * img_h)
        card_rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_card_w and h > min_card_h:
                card_rects.append((x, w))
        card_rects.sort(key=lambda r: r[0])
        merged = self._merge_overlapping(card_rects)
        if len(merged) <= 1:
            filtered = self._filter_by_arc(merged, img_w, img_h)
            merged = filtered if len(filtered) >= len(merged) else merged
        count = len(merged)
        if count < 1:
            self.logger.warning("Canny卡牌计数为0，返回0")
            return 0
        return count

    def _filter_by_arc(self, rects: List[Tuple[int, int, int, int]], img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        if len(rects) <= 1:
            return rects
        arc_cy_ratio = self.recog_cfg.get("card_arc_center_y_ratio", 0.42)
        arc_r_ratio = self.recog_cfg.get("card_arc_radius_ratio", 0.45)
        tolerance = self.recog_cfg.get("card_arc_tolerance", 0.03)

        arc_cx = img_w / 2
        arc_cy = img_h * arc_cy_ratio
        arc_r = img_w * arc_r_ratio
        tol_px = img_h * tolerance

        filtered = []
        for rx, ry, rw, rh in rects:
            card_cx = rx + rw / 2
            dist = np.sqrt((card_cx - arc_cx) ** 2)
            expected_y = arc_cy - arc_r + np.sqrt(max(0, arc_r ** 2 - dist ** 2))
            card_by = ry + rh
            if abs(card_by - expected_y) < tol_px * 3:
                filtered.append((rx, ry, rw, rh))

        if len(filtered) < max(1, len(rects) * 0.3):
            return rects
        return filtered

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
        try:
            x1 = max(0, cx)
            y1 = max(0, cy)
            x2 = min(img.shape[1], cx + cw)
            y2 = min(img.shape[0], cy + ch)
            if y2 <= y1 or x2 <= x1:
                return 0
            card = img[y1:y2, x1:x2]
            if self._cost_templates:
                return self._match_card_cost(card)
            return self._ocr_card_cost(card)
        except Exception:
            return 0

    def _match_card_cost(self, card: np.ndarray) -> int:
        cost, _ = self._match_card_cost_with_score(card)
        return cost

    @staticmethod
    def _rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
        h, w = template.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(template, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _get_rotate_angles(self) -> List[float]:
        return self.recog_cfg.get("template_rotate_angles",
                                   [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15])

    def _match_card_cost_with_score(self, card: np.ndarray) -> Tuple[int, float]:
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if len(card.shape) == 3 else card
        threshold = self.cfg["templates"].get("match_threshold", 0.70)
        angles = self._get_rotate_angles()
        scales = [0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]

        best_val = 0.0
        best_digit = 0

        # Pass 1: angle=0 only (fast path for center cards)
        for digit, tmpl in self._cost_templates.items():
            for sc in scales:
                sw = int(tmpl.shape[1] * sc)
                sh = int(tmpl.shape[0] * sc)
                if sw < 5 or sh < 5 or sw > gray.shape[1] or sh > gray.shape[0]:
                    continue
                st = cv2.resize(tmpl, (sw, sh))
                corr = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(corr)
                if max_val > best_val:
                    best_val = max_val
                    best_digit = digit

        if best_val >= threshold:
            return best_digit, best_val

        # Pass 2: rotated search for tilted edge cards
        for digit, tmpl in self._cost_templates.items():
            for sc in scales:
                for angle in angles:
                    if angle == 0.0:
                        continue
                    rotated = self._rotate_template(tmpl, angle)
                    rsw = int(rotated.shape[1] * sc)
                    rsh = int(rotated.shape[0] * sc)
                    if rsw < 5 or rsh < 5 or rsw > gray.shape[1] or rsh > gray.shape[0]:
                        continue
                    st = cv2.resize(rotated, (rsw, rsh))
                    corr = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(corr)
                    if max_val > best_val:
                        best_val = max_val
                        best_digit = digit

        # Always return best match (caller decides threshold)
        if best_digit > 0:
            return best_digit, best_val
        return 0, 0.0

    def _ocr_card_cost(self, gem: np.ndarray) -> int:
        cost = self._ocr_number(gem)
        if 0 <= cost <= 20:
            return cost
        return 0

    def _ocr_card_cost_improved(self, card_segment: np.ndarray) -> int:
        if card_segment is None or card_segment.size == 0:
            return 0

        best_tm_cost = 0
        best_tm_score = 0.0

        if self._cost_templates:
            best_tm_cost, best_tm_score = self._match_card_cost_with_score(card_segment)
            if best_tm_score >= self.cfg["templates"].get("match_threshold", 0.70):
                return best_tm_cost

        crystal_roi = self._extract_mana_crystal_roi(card_segment)
        if crystal_roi is not None and crystal_roi.size > 0:
            if self._cost_templates:
                cost, match_score = self._match_cost_on_crystal(crystal_roi)
                if cost > 0 and match_score >= 0.60:
                    return cost

            cost = self._ocr_mana_crystal_digit(crystal_roi)
            if 0 < cost <= 20:
                return cost

        cost = self._ocr_number(card_segment)
        if 0 < cost <= 20:
            return cost

        # Low-confidence fallback: use template match best guess
        if best_tm_score >= 0.65:
            self.logger.debug(f"费用OCR低置信度回退: {best_tm_cost} (score={best_tm_score:.3f})")
            return best_tm_cost

        return 0

    def _match_cost_on_crystal(self, crystal_roi: np.ndarray) -> Tuple[int, float]:
        gray = cv2.cvtColor(crystal_roi, cv2.COLOR_BGR2GRAY) if len(crystal_roi.shape) == 3 else crystal_roi
        threshold = self.cfg["templates"].get("match_threshold", 0.65)
        angles = self._get_rotate_angles()
        scales = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

        best_val = 0.0
        best_digit = 0

        # Pass 1: angle=0 only
        for digit, tmpl in self._cost_templates.items():
            for sc in scales:
                sw = int(tmpl.shape[1] * sc)
                sh = int(tmpl.shape[0] * sc)
                if sw < 5 or sh < 5 or sw > gray.shape[1] or sh > gray.shape[0]:
                    continue
                st = cv2.resize(tmpl, (sw, sh))
                corr = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(corr)
                if max_val > best_val:
                    best_val = max_val
                    best_digit = digit

        if best_val >= threshold:
            return best_digit, best_val

        # Pass 2: rotated search
        for digit, tmpl in self._cost_templates.items():
            for sc in scales:
                for angle in angles:
                    if angle == 0.0:
                        continue
                    rotated = self._rotate_template(tmpl, angle)
                    rsw = int(rotated.shape[1] * sc)
                    rsh = int(rotated.shape[0] * sc)
                    if rsw < 5 or rsh < 5 or rsw > gray.shape[1] or rsh > gray.shape[0]:
                        continue
                    st = cv2.resize(rotated, (rsw, rsh))
                    corr = cv2.matchTemplate(gray, st, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(corr)
                    if max_val > best_val:
                        best_val = max_val
                        best_digit = digit

        if best_val >= threshold:
            return best_digit, best_val
        return 0, 0.0

    def _extract_mana_crystal_roi(self, card_segment: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w = card_segment.shape[:2]
            x1 = max(0, int(w * 0.08))
            x2 = min(w, int(w * 0.80))
            y1 = 2
            y2 = min(h, int(h * 0.30))
            if x2 <= x1 or y2 <= y1:
                return None
            return card_segment[y1:y2, x1:x2]
        except Exception:
            return None

    def _ocr_mana_crystal_digit(self, crystal_roi: np.ndarray) -> int:
        return self._ocr_number(crystal_roi)

    def _detect_minions(self, crops: dict, full_img: np.ndarray) -> Tuple[List[MinionInfo], List[MinionInfo]]:
        our_minions = []
        opponent_minions = []
        try:
            our_zone = crops.get("our_board_region")
            if our_zone is not None and our_zone.size > 0:
                ox, oy, _, _ = self._game_region_pixels("our_board_region", full_img)
                our_minions = self._find_minions_in_zone(our_zone, oy, ox, full_img)

            opp_zone = crops.get("enemy_board_region")
            if opp_zone is not None and opp_zone.size > 0:
                px, py, _, _ = self._game_region_pixels("enemy_board_region", full_img)
                opponent_minions = self._find_minions_in_zone(opp_zone, py, px, full_img)

        except Exception as e:
            self.logger.error(f"随从识别失败: {e}")

        return our_minions, opponent_minions

    def _find_minions_in_zone(
        self, zone: np.ndarray, offset_y: int, offset_x: int, full_img: np.ndarray
    ) -> List[MinionInfo]:
        minions = []
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        cl = self.recog_cfg.get("minion_canny_lower", 50)
        cu = self.recog_cfg.get("minion_canny_upper", 150)
        edges = cv2.Canny(gray, cl, cu)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        border_lower = np.array(self.cfg["game"].get("minion_border_color_lower", [0, 20, 80]), dtype=np.uint8)
        border_upper = np.array(self.cfg["game"].get("minion_border_color_upper", [30, 255, 255]), dtype=np.uint8)
        border_mask = cv2.inRange(hsv, border_lower, border_upper)

        combined = cv2.bitwise_or(edges, cv2.Canny(border_mask, 50, 150))
        kernel2 = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel2, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = full_img.shape[:2]
        min_w = int(self.cfg["game"].get("minion_min_size", [0.016, 0.037])[0] * img_w)
        min_h = int(self.cfg["game"].get("minion_min_size", [0.016, 0.037])[1] * img_h)
        max_w = int(self.cfg["game"].get("minion_max_size", [0.104, 0.278])[0] * img_w)
        max_h = int(self.cfg["game"].get("minion_max_size", [0.104, 0.278])[1] * img_h)
        ocr_w = int(self.cfg["game"].get("minion_ocr_size", [0.03, 0.03])[0] * img_w)
        ocr_h = int(self.cfg["game"].get("minion_ocr_size", [0.03, 0.03])[1] * img_h)
        merge_gap = int(self.recog_cfg.get("board_gap_merge_ratio", 0.01) * img_w)

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
            attack, health = self._ocr_minion_numbers(zone, mx, my, mw, mh)
            if attack == 0 and health == 0:
                continue
            can_attack = self._detect_attackable_green_border(full_img, abs_x, abs_y, mw, mh)
            minions.append(
                MinionInfo(
                    health=health,
                    attack=attack,
                    position=(abs_x, abs_y, abs_x + mw, abs_y + mh),
                    can_attack=can_attack,
                )
            )

        return minions

    def _ocr_minion_numbers(self, zone: np.ndarray, mx: int, my: int, mw: int, mh: int) -> Tuple[int, int]:
        btm_fraction = 0.30
        btm_y = my + int(mh * (1.0 - btm_fraction))
        bottom = zone[btm_y: my + mh, mx: mx + mw]
        if bottom.size == 0:
            return 0, 0

        # OTSU first: handles white/red/green numbers regardless of color
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        for method in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
            _, thresh = cv2.threshold(gray, 0, 255, method)
            if cv2.countNonZero(thresh) >= 10:
                result = self._ocr_minion_from_mask(bottom, thresh)
                if result != (0, 0):
                    return result

        for _, lower, upper in [self._digit_color_masks[0], self._digit_color_masks[3]]:
            mask = cv2.inRange(bottom, lower, upper)
            if cv2.countNonZero(mask) >= 10:
                result = self._ocr_minion_from_mask(bottom, mask)
                if result != (0, 0):
                    return result

        combined_mask = np.zeros(bottom.shape[:2], dtype=np.uint8)
        for _, lower, upper in self._digit_color_masks:
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(bottom, lower, upper))
        if cv2.countNonZero(combined_mask) >= 10:
            return self._ocr_minion_from_mask(bottom, combined_mask)
        return 0, 0

    def _ocr_minion_from_mask(self, bottom: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mid_x = bottom.shape[1] // 2
        minion_h = bottom.shape[0]
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            lx, ly, lw, lh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            if area < 8 or lw < 3 or lh < 4:
                continue
            aspect = lh / max(1, lw)
            if aspect < 0.5 or aspect > 5.0:
                continue
            if lw > bottom.shape[1] * 0.5 or lh > minion_h * 0.7:
                continue

            cx = lx + lw // 2
            roi = bottom[ly: ly + lh, lx: lx + lw]
            pad = 4
            roi_p = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            val = self._ocr_single_digit(roi_p)
            if val > 0:
                candidates.append((cx, area, val))

        left_candidates = [(a, v) for cx, a, v in candidates if cx < mid_x]
        right_candidates = [(a, v) for cx, a, v in candidates if cx >= mid_x]
        attack = max(left_candidates, key=lambda x: x[0])[1] if left_candidates else 0
        health = max(right_candidates, key=lambda x: x[0])[1] if right_candidates else 0
        return attack, health

    def _ocr_single_digit(self, roi: np.ndarray) -> int:
        return self._ocr_number(roi)

        return 0

    def _detect_attackable_green_border(
        self, img: np.ndarray, minion_x: int, minion_y: int, minion_w: int, minion_h: int
    ) -> bool:
        try:
            border_w = self.cfg["game"].get("attackable_green_border_width", 6)
            green_lower = np.array(self.cfg["game"]["attackable_green_lower"], dtype=np.uint8)
            green_upper = np.array(self.cfg["game"]["attackable_green_upper"], dtype=np.uint8)
            min_ratio = self.cfg["game"].get("attackable_green_min_ratio", 0.1)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            edges = [
                (minion_x, minion_y, minion_x + minion_w, minion_y + border_w),
                (minion_x, minion_y + minion_h - border_w, minion_x + minion_w, minion_y + minion_h),
                (minion_x, minion_y, minion_x + border_w, minion_y + minion_h),
                (minion_x + minion_w - border_w, minion_y, minion_x + minion_w, minion_y + minion_h),
            ]

            total_checked = 0
            green_pixels = 0
            for bx1, by1, bx2, by2 in edges:
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(img.shape[1], bx2)
                by2 = min(img.shape[0], by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                roi = hsv[by1:by2, bx1:bx2]
                mask = cv2.inRange(roi, green_lower, green_upper)
                total_checked += mask.size
                green_pixels += cv2.countNonZero(mask)

            if total_checked == 0:
                return False
            ratio = green_pixels / total_checked
            return ratio >= min_ratio
        except Exception:
            return False

    def _detect_turn(self, turn_roi: np.ndarray) -> bool:
        key = self._detect_turn_key(turn_roi)
        if key:
            return key in ("end_turn_yellow", "end_turn_green")
        return False

    def _detect_turn_key(self, turn_roi: np.ndarray) -> Optional[str]:
        if turn_roi is None or turn_roi.size == 0:
            return None
        hsv = cv2.cvtColor(turn_roi, cv2.COLOR_BGR2HSV)
        total = turn_roi.shape[0] * turn_roi.shape[1]
        yellow_px = cv2.countNonZero(cv2.inRange(hsv, (18, 80, 120), (45, 255, 255)))
        green_px = cv2.countNonZero(cv2.inRange(hsv, (35, 60, 80), (95, 255, 255)))
        ratio = 0.03
        if yellow_px > total * ratio:
            return "end_turn_yellow"
        if green_px > total * ratio:
            return "end_turn_green"
        return "end_turn_opponent"

    def detect_end_turn_button(self, turn_roi: np.ndarray) -> bool:
        key = self._detect_turn_key(turn_roi)
        return key in ("end_turn_yellow", "end_turn_green")

    def detect_end_turn_green(self, turn_roi: np.ndarray) -> bool:
        key = self._detect_turn_key(turn_roi)
        return key == "end_turn_green"

    def find_end_turn_button_bbox(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return the center of the end turn button region from config as a clickable bbox."""
        region = self._game_region_pixels("end_turn_button_region", img)
        if not region or region[2] <= 0 or region[3] <= 0:
            return None
        cx = region[0] + region[2] // 2
        cy = region[1] + region[3] // 2
        return (cx - 10, cy - 10, cx + 10, cy + 10)

    def detect_game_over(self, crops: dict, turn_roi: np.ndarray, img: np.ndarray) -> bool:
        our_health = self._extract_health(crops.get("health_region"), "health_region")
        opp_health = self._extract_health(crops.get("enemy_health_region"), "enemy_health_region")
        has_btn = self.detect_end_turn_button(turn_roi)
        if our_health == 0 and opp_health == 0 and not has_btn:
            return True
        if not self.tesseract_available:
            return False
        try:
            h, w = img.shape[:2]
            center_roi = img[int(h * 0.35):int(h * 0.65), int(w * 0.3):int(w * 0.7)]
            text = self._safe_ocr(center_roi, config="--psm 6")
            text_lower = text.strip().lower()
            for kw in ["胜利", "失败", "victory", "defeat", "you win", "you lose", "经验", "exp"]:
                if kw in text_lower:
                    return True
        except Exception:
            pass
        return False

    def detect_main_menu(self, play_roi: np.ndarray) -> bool:
        if play_roi is None or play_roi.size == 0:
            return False
        found, key, _ = self._match_any_template(play_roi, ["play"])
        if found:
            return True
        if not self.tesseract_available:
            return False
        try:
            text = self._safe_ocr(play_roi, config="--psm 6")
            text_lower = text.strip().lower()
            for kw in ["play", "开始", "对战", "排位", "休闲", "hplay", "bplay",
                        "standard", "wild", "casual", "ranked"]:
                if kw in text_lower:
                    return True
        except Exception:
            pass
        return False

    def detect_post_game(self, crops: dict, turn_roi: np.ndarray, img: np.ndarray) -> bool:
        if self.detect_game_over(crops, turn_roi, img):
            return True
        our_health = self._extract_health(crops.get("health_region"), "health_region")
        opp_health = self._extract_health(crops.get("enemy_health_region"), "enemy_health_region")
        has_btn = self.detect_end_turn_button(turn_roi)
        if our_health == 0 and opp_health == 0 and not has_btn:
            return True
        return False

    def _consistent_value(self, cache: Deque, new_val: int) -> int:
        cache.append(new_val)
        if len(cache) < self._consistent_needed:
            return new_val
        recent = list(cache)[-self._consistent_needed:]
        from collections import Counter
        counter = Counter(recent)
        most_common_val, count = counter.most_common(1)[0]
        if count >= self._consistent_needed:
            return most_common_val
        else:
            return max(set(recent), key=recent.count)

    def _consistent_bool(self, cache: Deque, new_val: bool) -> bool:
        cache.append(new_val)
        if len(cache) < self._consistent_needed:
            return new_val
        true_count = sum(cache)
        false_count = len(cache) - true_count
        if true_count >= self._consistent_needed:
            return True
        if false_count >= self._consistent_needed:
            return False
        return new_val

    def _save_debug_crops(self, crops: dict, img: np.ndarray, game_state: GameState):
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

        for key, roi in crops.items():
            if roi is not None and roi.size > 0:
                _save(key, roi)

        for i, card in enumerate(game_state.hand_cards):
            x1, y1, x2, y2 = card.position
            if x2 > x1 and y2 > y1:
                _save(f"card_{i:02d}_cost{card.cost}", img[y1:y2, x1:x2])

        for i, m in enumerate(game_state.our_minions):
            x1, y1, x2, y2 = m.position
            if x2 > x1 and y2 > y1:
                tag = "atk" if m.can_attack else "idle"
                _save(f"our_minion_{i:02d}_{tag}", img[y1:y2, x1:x2])

        for i, m in enumerate(game_state.opponent_minions):
            x1, y1, x2, y2 = m.position
            if x2 > x1 and y2 > y1:
                _save(f"opp_minion_{i:02d}", img[y1:y2, x1:x2])

    def recognize(self) -> GameState:
        img = self.capture_screen()
        game_state = GameState(screenshot=img, timestamp=time.time())

        crops = self._crop_all_regions(img)
        img_hw = img.shape[:2]
        turn_roi = crops.get("turn_indicator_region")
        hand_roi = crops.get("hand_card_region")
        hand_offset = self._game_region_pixels("hand_card_region", img)

        if self._log_enabled:
            self._log_tracker.tick()
            log_state = self._log_tracker.get_state()
            if log_state.is_valid:
                self._apply_log_state(game_state, log_state)
            else:
                if self._log_tracker.available:
                    if time.time() - self._last_log_warning > 30:
                        self.logger.warning("日志监控已就绪但尚未捕获到游戏状态（可能对局未开始）")
                        self._last_log_warning = time.time()

        if not game_state.log_data_available:
            raw_turn = self._detect_turn(turn_roi)
            game_state.is_our_turn = self._consistent_bool(self._turn_cache, raw_turn)
            raw_health = self._extract_health(crops.get("health_region"), "health_region")
            game_state.our_health = self._consistent_value(self._health_cache, raw_health)
            raw_enemy_health = self._extract_health(crops.get("enemy_health_region"), "enemy_health_region")
            game_state.opponent_health = self._consistent_value(self._enemy_health_cache, raw_enemy_health)
            game_state.our_mana, game_state.total_mana = self._extract_mana(crops, img_hw)
            game_state.hand_cards = self._detect_hand_cards(hand_roi, img_hw, (hand_offset[0], hand_offset[1]))
            game_state.our_minions, game_state.opponent_minions = self._detect_minions(crops, img)

        if game_state.log_data_available:
            game_state.hand_cards = self._merge_log_hand_cards([])
            game_state.our_minions = self._merge_log_minions([], controller=1)
            game_state.opponent_minions = self._merge_log_minions([], controller=2)

        game_state.has_end_turn_button = self.detect_end_turn_button(turn_roi)
        game_state.is_end_turn_green = self.detect_end_turn_green(turn_roi)
        game_state.end_turn_button_bbox = self.find_end_turn_button_bbox(img)
        game_state.is_game_over = self.detect_game_over(crops, turn_roi, img)
        game_state.is_post_game = game_state.is_game_over
        game_state.is_main_menu = self.detect_main_menu(crops.get("play_button_region"))

        mana = game_state.our_mana
        for card in game_state.hand_cards:
            card.is_playable = card.cost <= mana and card.cost > 0

        hand_costs = [f"{c.cost}({'随' if c.card_type=='minion' else '法'})" for c in game_state.hand_cards]
        our_m_str = " ".join(f"{m.attack}/{m.health}" for m in game_state.our_minions) or "无"
        opp_m_str = " ".join(f"{m.attack}/{m.health}" for m in game_state.opponent_minions) or "无"
        log_tag = "[日志]" if game_state.log_data_available else "[CV]"
        self.logger.info(
            f"{'[我方回合]' if game_state.is_our_turn else '[敌方回合]'} {log_tag}"
            f"血量 {game_state.our_health}/{game_state.opponent_health} | "
            f"手牌{len(game_state.hand_cards)}张 [{', '.join(hand_costs)}] | "
            f"水晶 {game_state.our_mana}/{game_state.total_mana} | "
            f"我方随从 [{our_m_str}] | 敌方随从 [{opp_m_str}]"
        )

        self._save_debug_crops(crops, img, game_state)

        return game_state

    def _apply_log_state(self, game_state: GameState, log_state):
        """Apply log-derived state to game_state (100% accurate data)."""
        game_state.log_data_available = True
        game_state.is_our_turn = log_state.is_our_turn
        game_state.our_health = log_state.our_health
        game_state.opponent_health = log_state.opponent_health
        game_state.our_mana = log_state.our_mana
        game_state.total_mana = log_state.total_mana
        game_state.our_armor = log_state.our_armor
        game_state.opponent_armor = log_state.opponent_armor
        game_state.is_game_over = log_state.is_game_over
        game_state.hero_power_used = log_state.hero_power_used_this_turn
        game_state.action_options = log_state.action_options
        self._log_state = log_state

    def _get_config_position(self, key: str, index: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Get card/minion position from config by hand size / board slot index."""
        positions = self.cfg["game"].get(key, {})
        slot = positions.get(index)
        if slot is None:
            return (0, 0, 0, 0)
        x1 = int(slot[0] * img_w)
        y1 = int(slot[1] * img_h)
        x2 = int((slot[0] + slot[2]) * img_w)
        y2 = int((slot[1] + slot[3]) * img_h)
        return (x1, y1, x2, y2)

    def _merge_log_hand_cards(self, cv_cards: List[CardInfo]) -> List[CardInfo]:
        log_state = getattr(self, '_log_state', None)
        if not log_state or not log_state.hand_cards:
            return cv_cards
        log_cards = log_state.hand_cards
        hand_size = len(log_cards)
        gw = self.cfg["screen"]["game_region"]["width"]
        gh = self.cfg["screen"]["game_region"]["height"]
        result = []
        for i, lc in enumerate(log_cards):
            slot = (self.cfg["game"].get("hand_card_positions", {})
                    .get(hand_size, {})
                    .get(lc.zone_position - 1))  # zone_position is 1-based, config is 0-based
            if slot:
                pos = (int(slot[0] * gw), int(slot[1] * gh),
                       int((slot[0] + slot[2]) * gw), int((slot[1] + slot[3]) * gh))
            else:
                pos = (0, 0, 0, 0)
            result.append(CardInfo(
                name=lc.name, cost=lc.cost,
                position=pos,
                card_type=lc.card_type, card_id=lc.card_id,
                zone_position=lc.zone_position,
            ))
        return result

    def _merge_log_minions(self, cv_minions: List[MinionInfo], controller: int = 1) -> List[MinionInfo]:
        log_state = getattr(self, '_log_state', None)
        if not log_state:
            return cv_minions
        log_minions = log_state.our_minions if controller == 1 else log_state.opponent_minions
        gw = self.cfg["screen"]["game_region"]["width"]
        gh = self.cfg["screen"]["game_region"]["height"]
        board_key = "our_board_positions" if controller == 1 else "enemy_board_positions"
        result = []
        for i, lm in enumerate(log_minions):
            pos = self._get_config_position(board_key, lm.zone_position - 1, gw, gh)  # 1-based to 0-based
            if pos == (0, 0, 0, 0) and i < len(cv_minions):
                pos = cv_minions[i].position
            result.append(MinionInfo(
                health=lm.health, attack=lm.attack, position=pos,
                can_attack=lm.can_attack, name=lm.name, card_id=lm.card_id,
                has_taunt=lm.has_taunt, has_divine_shield=lm.has_divine_shield,
                has_stealth=lm.has_stealth, is_elusive=lm.is_elusive,
                has_poisonous=lm.has_poisonous, has_reborn=lm.has_reborn,
                has_deathrattle=lm.has_deathrattle, has_battlecry=lm.has_battlecry,
                has_lifesteal=lm.has_lifesteal, has_rush=lm.has_rush,
                has_charge=lm.has_charge, is_frozen=lm.is_frozen,
                is_silenced=lm.is_silenced, card_race=lm.card_race,
            ))
        return result
