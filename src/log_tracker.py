import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.logger import HsBatLogger
from src.log_config import get_power_log_path
from src.card_db import get_card_db


@dataclass
class LogCard:
    card_id: str = ""
    name: str = ""
    cost: int = 0
    card_type: str = "unknown"
    zone_position: int = 0
    entity_id: int = 0
    is_playable: bool = False


@dataclass
class LogMinion:
    card_id: str = ""
    name: str = ""
    attack: int = 0
    health: int = 0
    zone_position: int = 0
    entity_id: int = 0
    can_attack: bool = False
    can_attack_hero: bool = False
    has_taunt: bool = False
    has_divine_shield: bool = False
    has_stealth: bool = False
    has_windfury: bool = False
    has_lifesteal: bool = False
    has_rush: bool = False
    has_charge: bool = False
    is_frozen: bool = False
    is_silenced: bool = False
    is_elusive: bool = False
    has_poisonous: bool = False
    has_reborn: bool = False
    has_deathrattle: bool = False
    has_battlecry: bool = False
    card_race: str = ""


@dataclass
class LogHero:
    entity_id: int = 0
    card_id: str = ""
    health: int = 30
    armor: int = 0
    hero_power_cost: int = 2
    hero_power_used: bool = False


@dataclass
class LogActionTarget:
    entity_id: int = 0
    card_id: str = ""
    zone: str = ""
    is_valid: bool = True
    error: str = ""


@dataclass
class LogActionOption:
    option_index: int = 0
    action_type: str = ""          # END_TURN, POWER
    main_entity_id: int = 0
    main_card_id: str = ""
    main_zone: str = ""
    is_playable: bool = False      # error=NONE
    error: str = ""
    targets: List[LogActionTarget] = field(default_factory=list)


@dataclass
class LogState:
    is_valid: bool = False
    is_our_turn: bool = False
    our_health: int = 0
    opponent_health: int = 0
    our_mana: int = 0
    total_mana: int = 0
    our_armor: int = 0
    opponent_armor: int = 0
    turn_number: int = 0
    game_step: str = ""
    is_game_over: bool = False
    hand_cards: List[LogCard] = field(default_factory=list)
    our_minions: List[LogMinion] = field(default_factory=list)
    opponent_minions: List[LogMinion] = field(default_factory=list)
    our_hero: Optional[LogHero] = None
    opponent_hero: Optional[LogHero] = None
    timestamp: float = 0.0
    # Player entity counters
    corpses: int = 0
    combo_active: bool = False
    num_cards_played_this_turn: int = 0
    num_minions_played_this_turn: int = 0
    hero_power_used_this_turn: bool = False
    temp_resources: int = 0
    overload_locked: int = 0
    first_player: bool = False
    num_cards_drawn_this_turn: int = 0
    num_minions_killed_this_turn: int = 0
    # DebugPrintOptions
    action_options: List[LogActionOption] = field(default_factory=list)


class HearthstoneLogTracker:
    GAME_ENTITY_ID = 1

    # String -> int mappings for tag values
    ZONE_MAP = {"INVALID": 0, "PLAY": 1, "DECK": 2, "HAND": 3, "GRAVEYARD": 4,
                "REMOVEDFROMGAME": 5, "SECRET": 6, "SETASIDE": 7}
    CARDTYPE_MAP = {"INVALID": 0, "MINION": 1, "SPELL": 2, "HERO": 3,
                    "WEAPON": 4, "HERO_POWER": 5, "ENCHANTMENT": 6, "LOCATION": 7,
                    "PLAYER": 8}
    STEP_MAP = {"INVALID": 0, "BEGIN_MULLIGAN": 1, "MAIN_READY": 2, "MAIN_ACTION": 3,
                "MAIN_COMBAT": 4, "MAIN_END": 5, "FINAL_WRAPUP": 6, "FINAL_GAMEOVER": 7}
    STATE_MAP = {"INVALID": 0, "RUNNING": 1, "COMPLETE": 2, "CONCEDED": 3}

    # Tags we care about to reduce noise
    RELEVANT_TAGS = {
        "ZONE", "ZONE_POSITION", "CONTROLLER", "CARDTYPE", "COST", "ATK",
        "HEALTH", "DAMAGE", "ARMOR", "DURABILITY", "EXHAUSTED", "TAUNT",
        "DIVINE_SHIELD", "STEALTH", "WINDFURY", "NUM_ATTACKS_THIS_TURN",
        "FROZEN", "CANT_ATTACK", "CHARGE", "RUSH", "LIFESTEAL",
        "SILENCED", "CURRENT_PLAYER", "TURN", "STEP", "STATE",
        "RESOURCES", "RESOURCES_USED", "NUM_TURNS_IN_PLAY", "ENTITY_ID",
        "OVERLOAD_LOCKED", "CARD_ID", "CONTROLLER_PLAYER",
        "CANT_BE_TARGETED_BY_SPELLS", "CANT_BE_TARGETED_BY_HERO_POWERS",
        "POISONOUS", "REBORN", "DEATHRATTLE", "BATTLECRY",
        "CARDRACE", "CORPSES", "COMBO_ACTIVE", "NUM_CARDS_PLAYED_THIS_TURN",
        "NUM_MINIONS_PLAYED_THIS_TURN", "HEROPOWER_ACTIVATIONS_THIS_TURN",
        "TEMP_RESOURCES", "FIRST_PLAYER", "NUM_CARDS_DRAWN_THIS_TURN",
        "NUM_MINIONS_PLAYER_KILLED_THIS_TURN", "MAXRESOURCES",
        "NUM_RESOURCES_SPENT_THIS_GAME", "NUM_OPTIONS_PLAYED_THIS_TURN",
        "NUM_FRIENDLY_MINIONS_THAT_DIED_THIS_TURN",
    }

    def __init__(self, manual_path: str = None):
        self.logger = HsBatLogger().get_logger("LogTracker")
        self._log_path = get_power_log_path(manual_path)
        self._entities: Dict[int, Dict[str, int]] = {}   # eid -> {tag_name: int_value}
        self._card_ids: Dict[int, str] = {}               # eid -> CardID
        self._player_entity_id: Optional[int] = None
        self._opponent_entity_id: Optional[int] = None
        self._player_hero_id: Optional[int] = None
        self._opponent_hero_id: Optional[int] = None
        self._named_entities: Dict[str, int] = {}         # "玩家名#5299" -> eid
        self._player1_name: Optional[str] = None
        self._player2_name: Optional[str] = None
        self._last_file_pos = 0
        self._card_db = None
        self._available = False
        self._first_tick = True
        self._action_options: List[LogActionOption] = []
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self._log_path):
            self.logger.warning(f"Power.log 不存在: {self._log_path}")
            self._available = False
            return
        try:
            self._available = True
            file_size = os.path.getsize(self._log_path)
            self.logger.info(f"Power.log 监控已就绪: {self._log_path} (大小: {file_size})")
            # Start from position 0 on first tick to parse all existing data
            self._last_file_pos = 0
            self._first_tick = True
        except Exception as e:
            self.logger.warning(f"无法访问 Power.log: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def tick(self):
        if not self._available:
            return
        try:
            if not os.path.exists(self._log_path):
                return
            current_size = os.path.getsize(self._log_path)
            if current_size < self._last_file_pos:
                self.logger.info("Power.log 已轮转，重置状态")
                self._last_file_pos = 0
                self._entities.clear()
                self._card_ids.clear()
                self._named_entities.clear()
                self._player_entity_id = None
                self._opponent_entity_id = None
                self._player_hero_id = None
                self._opponent_hero_id = None
                self._first_tick = True
            if current_size > self._last_file_pos:
                with open(self._log_path, "r", encoding="utf-8") as f:
                    f.seek(self._last_file_pos)
                    new_data = f.read(current_size - self._last_file_pos)
                self._last_file_pos = current_size
                self._parse_lines(new_data)
            if self._first_tick and current_size > 0:
                self._first_tick = False
                if self._entities:
                    self.logger.info(f"全量解析完成: {len(self._entities)} 个实体, {len(self._card_ids)} 个卡牌ID")
        except Exception as e:
            self.logger.debug(f"日志读取异常: {e}")

    def _parse_lines(self, data: str):
        lines = data.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or not line.startswith("D "):
                i += 1
                continue
            try:
                i = self._parse_line_stateful(lines, i)
            except Exception:
                i += 1

    def _parse_line_stateful(self, lines: list, idx: int) -> int:
        line = lines[idx].strip()

        # Continuation lines: indented tag=value without event keyword
        if "tag=" in line and "value=" in line and "TAG_CHANGE" not in line and "FULL_ENTITY" not in line and "SHOW_ENTITY" not in line:
            if self._full_entity_id is not None:
                entity_id = self._full_entity_id
            else:
                # Try to find entity ID from previous FULL_ENTITY line
                entity_id = None
                for prev_idx in range(idx - 1, max(-1, idx - 30), -1):
                    pline = lines[prev_idx].strip()
                    if "FULL_ENTITY" in pline:
                        m = re.search(r"ID=(\d+)", pline)
                        if m:
                            entity_id = int(m.group(1))
                        break
                    if "SHOW_ENTITY" in pline:
                        m = re.search(r"id=(\d+)", pline)
                        if m:
                            entity_id = int(m.group(1))
                        break
                    if "TAG_CHANGE" in pline:
                        m = re.search(r"Entity=\[.*?id=(\d+)", pline)
                        if m:
                            entity_id = int(m.group(1))
                        break
                if entity_id is None:
                    return idx + 1
            self._parse_tag_pair(line, entity_id)
            return idx + 1

        if "CREATE_GAME" in line and "GameState.DebugPrintPower()" in line:
            self._entities.clear()
            self._card_ids.clear()
            self._player_entity_id = None
            self._opponent_entity_id = None
            self._player_hero_id = None
            self._opponent_hero_id = None
            self._action_options.clear()
            self._full_entity_id = None
            return idx + 1

        if "FULL_ENTITY" in line:
            self._full_entity_id = None
            # Match both uppercase ID= (Creating ID=64) and lowercase id= (in entity descriptors)
            id_match = re.search(r"\b[iI][dD]=(\d+)", line)
            card_match = re.search(r"[Cc]ard[Ii][Dd]=(\S+)", line)
            if id_match:
                entity_id = int(id_match.group(1))
                self._full_entity_id = entity_id
            else:
                # Indented FULL_ENTITY: entity info in brackets [entityName=... id=64 ...]
                m = re.search(r"id=(\d+)", line)
                if m:
                    entity_id = int(m.group(1))
                    self._full_entity_id = entity_id
            if card_match:
                card_id = card_match.group(1)
                if card_id:
                    self._card_ids[entity_id] = card_id
            # Also capture cardId= from entity descriptors (lowercase)
            cid_lower = re.search(r"cardId=(\S+)", line)
            if cid_lower and cid_lower.group(1):
                self._card_ids[entity_id] = cid_lower.group(1)
            if id_match or 'id=' in line:
                tag_pairs = re.findall(r"tag=(\S+)\s+value=(\S+)", line)
                for tag_name, value_str in tag_pairs:
                    if id_match:
                        self._apply_tag(entity_id, tag_name, value_str)
                    elif 'id=' in line:
                        self._apply_tag(entity_id, tag_name, value_str)
            return idx + 1

        if "TAG_CHANGE" in line:
            self._full_entity_id = None
            entity_id = self._extract_entity_id(line)
            if entity_id is not None:
                m = re.search(r"tag=(\S+)\s+value=(\S+)", line)
                if m:
                    self._apply_tag(entity_id, m.group(1), m.group(2))
                cid_match = re.search(r"cardId=(\S+)", line)
                if cid_match and cid_match.group(1):
                    self._card_ids[entity_id] = cid_match.group(1)
            return idx + 1

        if "SHOW_ENTITY" in line:
            id_match = re.search(r"id=(\d+)", line)
            card_match = re.search(r"cardId=(\S+)", line)
            if id_match:
                entity_id = int(id_match.group(1))
                self._full_entity_id = entity_id
                if card_match and card_match.group(1):
                    self._card_ids[entity_id] = card_match.group(1)
                tag_pairs = re.findall(r"tag=(\S+)\s+value=(\S+)", line)
                for tag_name, value_str in tag_pairs:
                    self._apply_tag(entity_id, tag_name, value_str)
            return idx + 1

        # Player EntityID=2 PlayerID=1 (player entity creation)
        if "Player EntityID=" in line:
            m = re.search(r"EntityID=(\d+)", line)
            if m:
                self._full_entity_id = int(m.group(1))
            return idx + 1

        # DebugPrintGame: PlayerID=1, PlayerName=玩家名
        if "DebugPrintGame()" in line and "PlayerID" in line:
            m = re.search(r"PlayerID=(\d+),\s*PlayerName=(\S+)", line)
            if m:
                pid = int(m.group(1))
                name = m.group(2)
                if pid == 1:
                    self._player1_name = name
                    self._named_entities[name] = 2
                elif pid == 2:
                    self._player2_name = name
                    self._named_entities[name] = 3
            return idx + 1

        # DebugPrintOptions: available actions with target validation
        # Only trigger on the block header line: DebugPrintOptions() - id=N
        if re.search(r"DebugPrintOptions\(\)\s*-\s*id=\d+", line):
            return self._parse_debug_print_options(lines, idx)

        return idx + 1

        return idx + 1

    def _parse_debug_print_options(self, lines: list, idx: int) -> int:
        options = []
        current_option = None
        i = idx
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("D "):
                i += 1
                continue
            if re.search(r"DebugPrintOptions\(\)\s*-\s*id=\d+", line) and i != idx:
                break
            if any(kw in line for kw in ("FULL_ENTITY", "TAG_CHANGE", "SHOW_ENTITY",
                                           "CREATE_GAME", "BLOCK_START", "BLOCK_END",
                                           "META_DATA", "SendOption", "DebugPrintPowerList")):
                break
            if "id=" in line and i == idx:
                i += 1
                continue
            if "option " in line:
                if current_option:
                    options.append(current_option)
                current_option = self._parse_option_line(line)
                i += 1
                continue
            if "target " in line and current_option is not None:
                target = self._parse_target_line(line)
                if target:
                    current_option.targets.append(target)
                i += 1
                continue
            i += 1
        if current_option:
            options.append(current_option)
        if options:
            self._action_options = options
        return i  # Return end index

    @staticmethod
    def _parse_option_line(line: str) -> Optional[LogActionOption]:
        m = re.search(r"option\s+(\d+)\s+type=(\S+)", line)
        if not m:
            return None
        opt_idx = int(m.group(1))
        action_type = m.group(2)
        main_entity_id = 0
        main_card_id = ""
        main_zone = ""
        em = re.search(r"id=(\d+)", line)
        if em:
            main_entity_id = int(em.group(1))
        cm = re.search(r"cardId=(\S+)", line)
        if cm:
            main_card_id = cm.group(1)
        zm = re.search(r"zone=(\w+)", line)
        if zm:
            main_zone = zm.group(1)
        em2 = re.search(r"error=(\S+)", line)
        error = em2.group(1) if em2 else ""
        is_playable = error == "NONE"
        return LogActionOption(
            option_index=opt_idx, action_type=action_type,
            main_entity_id=main_entity_id, main_card_id=main_card_id,
            main_zone=main_zone, is_playable=is_playable, error=error,
        )

    @staticmethod
    def _parse_target_line(line: str) -> Optional[LogActionTarget]:
        m = re.search(r"target\s+\d+\s+entity=", line)
        if not m:
            return None
        eid = 0
        em = re.search(r"id=(\d+)", line)
        if em:
            eid = int(em.group(1))
        card_id = ""
        cm = re.search(r"cardId=(\S+)", line)
        if cm:
            card_id = cm.group(1)
        zone = ""
        zm = re.search(r"zone=(\w+)", line)
        if zm:
            zone = zm.group(1)
        error = ""
        em2 = re.search(r"error=(\S+)", line)
        if em2:
            error = em2.group(1)
        return LogActionTarget(
            entity_id=eid, card_id=card_id, zone=zone,
            is_valid=(error == "NONE"), error=error,
        )

    def _parse_tag_pair(self, line: str, entity_id: int):
        m = re.search(r"tag=(\S+)\s+value=(\S+)", line)
        if m:
            self._apply_tag(entity_id, m.group(1), m.group(2))

    def _extract_entity_id(self, line: str) -> Optional[int]:
        """Extract entity_id from a TAG_CHANGE line, handling various formats."""
        # Format 1: Entity=[entityName=... id=113 zone=PLAY ...]
        m = re.search(r"Entity=\[.*?id=(\d+)", line)
        if m:
            eid = int(m.group(1))
            player_match = re.search(r"player=(\d+)", line)
            if player_match and int(player_match.group(1)) in (1, 2):
                name_match = re.search(r"entityName=([^\]\s]+)", line)
                if name_match:
                    self._named_entities[name_match.group(1)] = eid
            return eid

        # Format 2: Entity=GameEntity
        if "Entity=GameEntity" in line:
            return 1

        # Format 3: Entity=数字 (player entity by ID)
        m = re.search(r"Entity=(\d+)\s+tag=", line)
        if m:
            eid = int(m.group(1))
            # Map player entities to their names
            if eid == 2 and self._player1_name and self._player1_name not in self._named_entities:
                self._named_entities[self._player1_name] = eid
            elif eid == 3 and self._player2_name and self._player2_name not in self._named_entities:
                self._named_entities[self._player2_name] = eid
            return eid

        # Format 4: Entity=玩家名#数字 (player entity by name)
        m = re.search(r"Entity=(\S+?)(?:#\d+)?\s+tag=", line)
        if m:
            name = m.group(1)
            # Build full name from the line (includes battletag)
            full_m = re.search(r"Entity=(\S+)\s+tag=", line)
            full_name = full_m.group(1) if full_m else name
            if full_name in self._named_entities:
                return self._named_entities[full_name]
            if name in self._named_entities:
                return self._named_entities[name]
            for n, eid in self._named_entities.items():
                if n.startswith(name):
                    return eid
            return 1

        return None

    def _apply_tag(self, entity_id: int, tag_name: str, value_str: str):
        """Apply a tag to an entity, mapping string values to ints where needed."""
        if entity_id <= 0:
            return

        # Only track relevant tags to avoid memory bloat
        if tag_name.isdigit():
            return  # Skip numeric tag IDs, keep only named tags

        if tag_name not in self.RELEVANT_TAGS:
            # Still allow some through that we explicitly handle
            pass  # Let everything through for now

        if entity_id not in self._entities:
            self._entities[entity_id] = {}

        # Parse value: could be int or string enum
        value = self._parse_value(tag_name, value_str)
        self._entities[entity_id][tag_name] = value

        # Track hero entities for later identification
        if tag_name == "CONTROLLER" and value in (1, 2):
            card_type = self._entities[entity_id].get("CARDTYPE", 0)
            if card_type == 3:
                if value == 1:
                    self.logger.debug(f"Found player hero: entity_id={entity_id}")
                else:
                    self.logger.debug(f"Found opponent hero: entity_id={entity_id}")

    def _parse_value(self, tag_name: str, value_str: str) -> int:
        """Convert a tag value string to an integer."""
        if value_str.isdigit():
            return int(value_str)

        # Try zone mapping
        if tag_name == "ZONE":
            return self.ZONE_MAP.get(value_str.upper(), 0)

        # Try card type mapping
        if tag_name in ("CARDTYPE",):
            return self.CARDTYPE_MAP.get(value_str.upper(), 0)

        # Try step mapping
        if tag_name == "STEP":
            return self.STEP_MAP.get(value_str.upper(), 0)

        # Try state mapping
        if tag_name == "STATE":
            return self.STATE_MAP.get(value_str.upper(), 0)

        # Boolean-like string values
        if value_str.upper() in ("TRUE", "YES"):
            return 1
        if value_str.upper() in ("FALSE", "NO", "NONE"):
            return 0

        # Fallback: try parsing as int, otherwise 0
        try:
            return int(value_str)
        except ValueError:
            return 0

    def get_state(self) -> LogState:
        state = LogState(timestamp=time.time())
        if not self._available or not self._entities:
            return state

        game_entity = self._entities.get(1, {})
        has_data = bool(game_entity) or len(self._card_ids) > 0
        if not has_data:
            return state

        state.is_valid = True
        state.turn_number = game_entity.get("TURN", 0)
        step_val = game_entity.get("STEP", 0)
        state.game_step = self._step_name(step_val)
        state_val = game_entity.get("STATE", 0)
        state.is_game_over = state_val == 2

        self._identify_players()
        if not self._player_entity_id and not self._player_hero_id:
            return state  # Can't identify players yet

        if self._player_entity_id:
            player_entity = self._entities.get(self._player_entity_id, {})
            state.is_our_turn = player_entity.get("CURRENT_PLAYER", 0) == 1
            p_resources = player_entity.get("RESOURCES", game_entity.get("RESOURCES", 0))
            p_used = player_entity.get("RESOURCES_USED", game_entity.get("RESOURCES_USED", 0))
            state.our_mana = max(0, p_resources - p_used)
            state.total_mana = max(p_resources, min(state.turn_number, 10))
        else:
            state.is_our_turn = state.turn_number > 0
            g_resources = game_entity.get("RESOURCES", 0)
            g_used = game_entity.get("RESOURCES_USED", 0)
            state.our_mana = max(0, g_resources - g_used)
            state.total_mana = max(g_resources, min(state.turn_number, 10))

        state.hand_cards = self._get_hand_cards()
        state.our_minions = self._get_board_minions(1)
        state.opponent_minions = self._get_board_minions(2)
        state.our_hero = self._get_hero(1)
        state.opponent_hero = self._get_hero(2)
        state.action_options = self._action_options

        # Player entity counters
        for eid in [self._player_entity_id, 2]:
            if eid and eid in self._entities:
                pe = self._entities[eid]
                state.corpses = pe.get("CORPSES", state.corpses)
                state.combo_active = pe.get("COMBO_ACTIVE", 0) == 1 or state.combo_active
                state.num_cards_played_this_turn = pe.get("NUM_CARDS_PLAYED_THIS_TURN", state.num_cards_played_this_turn)
                state.num_minions_played_this_turn = pe.get("NUM_MINIONS_PLAYED_THIS_TURN", state.num_minions_played_this_turn)
                state.hero_power_used_this_turn = pe.get("HEROPOWER_ACTIVATIONS_THIS_TURN", 0) > 0 or state.hero_power_used_this_turn
                state.temp_resources = pe.get("TEMP_RESOURCES", state.temp_resources)
                state.overload_locked = pe.get("OVERLOAD_LOCKED", state.overload_locked)
                state.first_player = pe.get("FIRST_PLAYER", 0) == 1 or state.first_player
                state.num_cards_drawn_this_turn = pe.get("NUM_CARDS_DRAWN_THIS_TURN", state.num_cards_drawn_this_turn)
                state.num_minions_killed_this_turn = pe.get("NUM_MINIONS_PLAYER_KILLED_THIS_TURN", state.num_minions_killed_this_turn)

        if state.our_hero:
            state.our_health = state.our_hero.health
            state.our_armor = state.our_hero.armor
        elif self._player_hero_id:
            hero = self._entities.get(self._player_hero_id, {})
            state.our_health = hero.get("HEALTH", 30) - hero.get("DAMAGE", 0)
            state.our_armor = hero.get("ARMOR", 0)
        if state.opponent_hero:
            state.opponent_health = state.opponent_hero.health
            state.opponent_armor = state.opponent_hero.armor
        elif self._opponent_hero_id:
            hero = self._entities.get(self._opponent_hero_id, {})
            state.opponent_health = hero.get("HEALTH", 30) - hero.get("DAMAGE", 0)
            state.opponent_armor = hero.get("ARMOR", 0)

        return state

    def _identify_players(self):
        # Player entities are always entity 2 (pid=1) and entity 3 (pid=2)
        if 2 in self._entities:
            self._player_entity_id = 2
        if 3 in self._entities:
            self._opponent_entity_id = 3

        # Also find hero entities for health/armor lookup
        for eid, tags in self._entities.items():
            controller = tags.get("CONTROLLER", 0)
            card_type = tags.get("CARDTYPE", 0)
            zone = tags.get("ZONE", 0)
            if controller == 1 and card_type == 3 and zone == 1:
                if self._player_hero_id is None:
                    self._player_hero_id = eid
            elif controller == 2 and card_type == 3 and zone == 1:
                if self._opponent_hero_id is None:
                    self._opponent_hero_id = eid

    def _get_hand_cards(self) -> List[LogCard]:
        cards = []
        db = self._get_card_db()
        for eid, tags in self._entities.items():
            controller = tags.get("CONTROLLER", 0)
            zone = tags.get("ZONE", 0)
            if controller != 1 or zone != 3:
                continue
            card_type_val = tags.get("CARDTYPE", 0)
            cost = tags.get("COST", 0)
            zone_pos = tags.get("ZONE_POSITION", 0)
            card_id = self._card_ids.get(eid, "")
            name = db.get_name(card_id) if card_id else ""
            card_type_str = self._card_type_name(card_type_val)
            cards.append(LogCard(
                card_id=card_id, name=name, cost=cost,
                card_type=card_type_str, zone_position=zone_pos,
                entity_id=eid,
            ))
        cards.sort(key=lambda c: c.zone_position)
        return cards

    def _get_board_minions(self, controller: int) -> List[LogMinion]:
        minions = []
        db = self._get_card_db()
        for eid, tags in self._entities.items():
            ent_controller = tags.get("CONTROLLER", 0)
            zone = tags.get("ZONE", 0)
            card_type = tags.get("CARDTYPE", 0)
            if ent_controller != controller or zone != 1 or card_type != 1:
                continue
            attack = tags.get("ATK", 0)
            health = tags.get("HEALTH", 0)
            zone_pos = tags.get("ZONE_POSITION", 0)
            exhausted = tags.get("EXHAUSTED", 0)
            frozen = tags.get("FROZEN", 0)
            cant_attack = tags.get("CANT_ATTACK", 0)
            num_attacks = tags.get("NUM_ATTACKS_THIS_TURN", 0)
            windfury = tags.get("WINDFURY", 1)
            charge = tags.get("CHARGE", 0)
            rush = tags.get("RUSH", 0)
            num_turns = tags.get("NUM_TURNS_IN_PLAY", 0)
            just_played = tags.get("JUST_PLAYED", 0)
            # 0-attack minions cannot attack at all
            if attack == 0:
                can_attack = False
                can_attack_hero = False
            elif just_played == 1 and charge == 0 and rush == 0:
                # Summoning sickness: no Charge/Rush -> cannot attack
                can_attack = False
                can_attack_hero = False
            elif just_played == 1 and rush == 1:
                # Rush: can attack minions but NOT hero on the first turn
                can_attack = (exhausted == 0 and frozen == 0 and cant_attack == 0
                             and num_attacks < max(1, windfury))
                can_attack_hero = False
            else:
                # Charge or already waited a turn: can attack anything
                can_attack = (exhausted == 0 and frozen == 0 and cant_attack == 0
                             and num_attacks < max(1, windfury))
                can_attack_hero = can_attack
            card_id = self._card_ids.get(eid, "")
            name = db.get_name(card_id) if card_id else ""
            race_val = tags.get("CARDRACE", 0)
            race_map = {0: "", 1: "BLOODELF", 2: "DRAENEI", 3: "DWARF", 4: "GNOME",
                       5: "GOBLIN", 6: "HUMAN", 7: "NIGHTELF", 8: "ORC", 9: "TAUREN",
                       10: "TROLL", 11: "UNDEAD", 12: "WORGEN", 13: "GOBLIN2",
                       14: "MURLOC", 15: "DEMON", 16: "SCOURGE", 17: "MECH",
                       18: "ELEMENTAL", 19: "OGRE", 20: "BEAST", 21: "TOTEM",
                       22: "NERUBIAN", 23: "PIRATE", 24: "DRAGON",
                       26: "QUILBOAR", 27: "NAGA"}
            minions.append(LogMinion(
                card_id=card_id, name=name, attack=attack, health=health,
                zone_position=zone_pos, entity_id=eid, can_attack=can_attack,
                can_attack_hero=can_attack_hero,
                has_taunt=tags.get("TAUNT", 0) == 1,
                has_divine_shield=tags.get("DIVINE_SHIELD", 0) == 1,
                has_stealth=tags.get("STEALTH", 0) == 1,
                has_windfury=windfury > 1,
                has_lifesteal=tags.get("LIFESTEAL", 0) == 1,
                has_rush=rush == 1,
                has_charge=charge == 1,
                is_frozen=frozen == 1,
                is_silenced=tags.get("SILENCED", 0) == 1,
                is_elusive=(tags.get("CANT_BE_TARGETED_BY_SPELLS", 0) == 1 or
                           tags.get("CANT_BE_TARGETED_BY_HERO_POWERS", 0) == 1),
                has_poisonous=tags.get("POISONOUS", 0) == 1,
                has_reborn=tags.get("REBORN", 0) == 1,
                has_deathrattle=tags.get("DEATHRATTLE", 0) == 1,
                has_battlecry=tags.get("BATTLECRY", 0) == 1,
                card_race=race_map.get(race_val, str(race_val)),
            ))
        minions.sort(key=lambda m: m.zone_position)
        return minions

    def _get_hero(self, controller: int) -> Optional[LogHero]:
        for eid, tags in self._entities.items():
            ent_controller = tags.get("CONTROLLER", 0)
            zone = tags.get("ZONE", 0)
            card_type = tags.get("CARDTYPE", 0)
            if ent_controller != controller or zone != 1 or card_type != 3:
                continue
            health = tags.get("HEALTH", 30) - tags.get("DAMAGE", 0)
            armor = tags.get("ARMOR", 0)
            card_id = self._card_ids.get(eid, "")
            hero_power_cost = 2
            hero_power_used = False
            for hp_eid, hp_tags in self._entities.items():
                hp_controller = hp_tags.get("CONTROLLER", 0)
                hp_zone = hp_tags.get("ZONE", 0)
                hp_type = hp_tags.get("CARDTYPE", 0)
                if hp_controller == controller and hp_zone == 1 and hp_type == 5:
                    hero_power_cost = hp_tags.get("COST", 2)
                    hero_power_used = hp_tags.get("EXHAUSTED", 0) == 1
                    break
            return LogHero(
                entity_id=eid, card_id=card_id, health=health, armor=armor,
                hero_power_cost=hero_power_cost, hero_power_used=hero_power_used,
            )
        return None

    def _get_card_db(self):
        if self._card_db is None:
            self._card_db = get_card_db()
        return self._card_db

    @staticmethod
    def _card_type_name(val: int) -> str:
        mapping = {0: "unknown", 1: "minion", 2: "spell", 3: "hero", 4: "weapon", 5: "hero_power"}
        return mapping.get(val, "unknown")

    @staticmethod
    def _step_name(val: int) -> str:
        mapping = {0: "INVALID", 1: "BEGIN_MULLIGAN", 2: "MAIN_READY",
                   3: "MAIN_ACTION", 4: "MAIN_COMBAT", 5: "MAIN_END",
                   6: "FINAL_WRAPUP", 7: "FINAL_GAMEOVER"}
        return mapping.get(val, f"STEP_{val}")
