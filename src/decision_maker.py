import json
import os
import time
from typing import Dict, List, Optional

import requests

from src.logger import HsBatLogger
from src.state_recognizer import GameState, MinionInfo


class DecisionMaker:
    def __init__(self, config: dict):
        self.cfg = config
        self.logger = HsBatLogger().get_logger("DecisionMaker")
        llm_cfg = config["llm"]
        self.llm_enabled = llm_cfg.get("enabled", False)
        self.api_key = llm_cfg.get("api_key", "") or os.environ.get("HSBAT_LLM_API_KEY", "")
        self.api_base = llm_cfg.get("api_base", "https://api.openai.com/v1")
        self.model = llm_cfg.get("model", "gpt-4o")
        self.timeout = llm_cfg.get("timeout", 15)
        self.max_retries = llm_cfg.get("max_retries", 2)
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.system_prompt = llm_cfg.get("system_prompt", "")

        self.memory: List[Dict] = []
        self.max_memory_len = 5

    def decide(self, game_state: GameState) -> Dict:
        if self.llm_enabled and self.api_key:
            return self._llm_decide(game_state)
        else:
            return self._rule_decide(game_state)

    # ================================================================
    #  规则引擎 (可配置策略，零延迟)
    # ================================================================
    def _rule_decide(self, game_state: GameState) -> Dict:
        if not game_state.is_our_turn:
            return {"action": "wait", "reason": "等待我方回合"}

        rule_cfg = self.cfg.get("rule_engine", {})
        play_strategy = rule_cfg.get("play_card_strategy", "high_cost_first")
        use_hp = rule_cfg.get("use_hero_power", True)

        # 1) 出牌
        playable = [c for c in game_state.hand_cards if c.is_playable]
        if playable:
            # Check if END_TURN is the only valid option (from log)
            if game_state.action_options:
                end_turn_opt = game_state.action_options[0]
                if end_turn_opt.is_playable:
                    self.logger.info("规则决策: 无可出牌, 结束回合")
                    return {
                        "action": "end_turn", "card_index": None, "target_index": None,
                        "target_type": None, "reason": "规则引擎: 无可操作(日志判定)",
                    }
            if play_strategy == "high_cost_first":
                card = max(playable, key=lambda c: c.cost)
            else:
                card = min(playable, key=lambda c: c.cost)
            self.logger.info(f"规则决策: 出牌 卡{playable.index(card)} ({card.name or '卡牌'} 费用{card.cost}, 类型{card.card_type}, 策略={play_strategy})")
            return {
                "action": "play_card",
                "card_index": game_state.hand_cards.index(card),
                "target_index": None,
                "target_type": None,
                "reason": f"规则引擎: 出牌({card.name or '卡牌'} 费{card.cost})",
            }

        # 2) 随从攻击
        attackable = [m for m in game_state.our_minions if m.can_attack]
        if attackable:
            decision = self._decide_attack_target(game_state, attackable)
            if decision:
                return decision

        # 3) 英雄技能
        if use_hp and game_state.our_mana >= 2:
            self.logger.info("规则决策: 使用英雄技能")
            return {
                "action": "use_hero_power",
                "card_index": None,
                "target_index": None,
                "target_type": None,
                "reason": "规则引擎: 使用英雄技能(剩余{mana}费)".format(mana=game_state.our_mana),
            }

        self.logger.info("规则决策: 结束回合")
        return {"action": "end_turn", "card_index": None, "target_index": None,
                "target_type": None, "reason": "规则引擎: 无操作可做"}

    def _decide_attack_target(
        self, game_state: GameState, attackable: List[MinionInfo]
    ) -> Optional[Dict]:
        rule_cfg = self.cfg.get("rule_engine", {})
        attack_strategy = rule_cfg.get("attack_strategy", "smart")
        defend_lethal = rule_cfg.get("defend_when_lethal", True)
        lethal_margin = rule_cfg.get("lethal_margin", 2)
        opponent_minions = game_state.opponent_minions

        # --- 无敌方随从：直接打脸 ---
        if not opponent_minions:
            self.logger.info("规则决策: 随从攻击敌方英雄")
            return {
                "action": "attack", "card_index": None,
                "target_index": None, "target_type": "enemy_hero",
                "reason": "规则引擎: 攻击敌方英雄",
            }

        # --- face_only 策略 ---
        if attack_strategy == "face_only":
            self.logger.info("规则决策: 攻击敌方英雄 (face_only)")
            return {
                "action": "attack", "card_index": None,
                "target_index": None, "target_type": "enemy_hero",
                "reason": "规则引擎: face_only 攻击英雄",
            }

        # --- trade_only 策略 ---
        if attack_strategy == "trade_only":
            best_target = self._pick_best_trade(attackable, opponent_minions)
            if best_target is not None:
                self.logger.info(f"规则决策: 解场 随从[{best_target}] (trade_only)")
                return {
                    "action": "attack", "card_index": None,
                    "target_index": best_target, "target_type": "enemy_minion",
                    "reason": f"规则引擎: trade_only 解场[{best_target}]",
                }

        # --- smart 策略 ---
        # 计算敌方场攻
        enemy_board_damage = sum(m.attack for m in opponent_minions)
        our_health = game_state.our_health

        # 计算我方场攻（仅可攻击的）
        our_board_damage = sum(m.attack for m in attackable)
        enemy_health = game_state.opponent_health

        # 危险判断：敌方场攻 > 我方血量 + 余量
        in_danger = defend_lethal and (enemy_board_damage >= our_health + lethal_margin)

        # 斩杀判断：我方场攻 >= 敌方血量
        has_lethal_on_enemy = our_board_damage >= enemy_health

        if in_danger:
            best_target = self._pick_best_trade(attackable, opponent_minions)
            self.logger.info(
                f"规则决策: 危险! 敌方场攻={enemy_board_damage} vs 血量={our_health}，优先解场"
            )
            return {
                "action": "attack", "card_index": None,
                "target_index": best_target, "target_type": "enemy_minion",
                "reason": f"规则引擎: 危险解场 [{enemy_board_damage}>{our_health}]",
            }

        if has_lethal_on_enemy:
            self.logger.info(
                f"规则决策: 斩杀! 场攻={our_board_damage} >= 敌血={enemy_health}"
            )
            return {
                "action": "attack", "card_index": None,
                "target_index": None, "target_type": "enemy_hero",
                "reason": f"规则引擎: 斩杀 场攻{our_board_damage}>={enemy_health}",
            }

        # 常规：聪明地选择目标
        # 如果有高威胁随从（高攻击力），优先解
        high_threat = [m for m in opponent_minions if m.attack >= 4]
        if high_threat:
            best_target = self._pick_best_trade(attackable, opponent_minions,
                                                 prefer_high_attack=True)
            self.logger.info(f"规则决策: 解高威胁随从")
            return {
                "action": "attack", "card_index": None,
                "target_index": best_target, "target_type": "enemy_minion",
                "reason": "规则引擎: 解高威胁随从",
            }

        # 无威胁 → 打脸
        self.logger.info("规则决策: 攻击敌方英雄")
        return {
            "action": "attack", "card_index": None,
            "target_index": None, "target_type": "enemy_hero",
            "reason": "规则引擎: 攻击英雄(无敌方威胁)",
        }

    def _pick_best_trade(
        self,
        attackers: List[MinionInfo],
        defenders: List[MinionInfo],
        prefer_high_attack: bool = False,
    ) -> Optional[int]:
        """选择最优解场目标，返回敌方随从在列表中的索引。"""
        if not defenders:
            return None

        if prefer_high_attack:
            # 优先打攻击力最高的
            best_idx = 0
            best_attack = -1
            for i, m in enumerate(defenders):
                if m.attack > best_attack:
                    best_attack = m.attack
                    best_idx = i
            return best_idx

        # 常规：找能白吃的（用攻击力大于敌方血量的去打），否则打最弱的
        attackers_sorted = sorted(attackers, key=lambda m: m.attack)
        defenders_indexed = [(i, m) for i, m in enumerate(defenders)]
        defenders_sorted = sorted(defenders_indexed, key=lambda x: x[1].health)

        for di, defender in defenders_sorted:
            for attacker in attackers_sorted:
                if attacker.attack >= defender.health:
                    return di

        return defenders_sorted[0][0]

    def build_attack_plan(self, game_state: GameState) -> List[Dict]:
        orders = []
        rule_cfg = self.cfg.get("rule_engine", {})
        attack_strategy = rule_cfg.get("attack_strategy", "smart")
        defend_lethal = rule_cfg.get("defend_when_lethal", True)
        lethal_margin = rule_cfg.get("lethal_margin", 2)

        attackable = [m for m in game_state.our_minions if m.can_attack]
        if not attackable:
            return orders

        opponent_minions = list(game_state.opponent_minions)

        # 先打脸策略
        if attack_strategy == "face_only" or not opponent_minions:
            for i, _ in enumerate(attackable):
                orders.append({"attacker_index": i, "target_index": None, "target_type": "enemy_hero"})
            return orders

        # 计算危险度
        enemy_board_damage = sum(m.attack for m in opponent_minions)
        in_danger = defend_lethal and (enemy_board_damage >= game_state.our_health + lethal_margin)
        our_board_damage = sum(m.attack for m in attackable)
        has_lethal = our_board_damage >= game_state.opponent_health

        # 斩杀 → 全打脸
        if has_lethal and not in_danger:
            for i, _ in enumerate(attackable):
                orders.append({"attacker_index": i, "target_index": None, "target_type": "enemy_hero"})
            return orders

        # 解场 + 打脸混合
        remaining_attackers = list(range(len(attackable)))
        remaining_defenders = list(range(len(opponent_minions)))

        # 每轮找一个最优交换
        while remaining_attackers and remaining_defenders:
            best_pair = None
            best_efficiency = float("inf")

            for ai in remaining_attackers:
                a = attackable[ai]
                for di in remaining_defenders:
                    d = opponent_minions[di]
                    # 效率 = 对攻击者的伤害 / 消灭的敌方攻击力
                    damage_to_attacker = max(0, d.attack - a.health) if d.attack > 0 else 0
                    if d.health <= a.attack:
                        efficiency = damage_to_attacker / max(d.attack, 1)
                        if efficiency < best_efficiency:
                            best_efficiency = efficiency
                            best_pair = (ai, di)

            if best_pair is None:
                break

            ai, di = best_pair
            orders.append({"attacker_index": ai, "target_index": di, "target_type": "enemy_minion"})
            remaining_attackers.remove(ai)
            remaining_defenders.remove(di)

        # 剩余攻击者打脸
        for ai in remaining_attackers:
            orders.append({"attacker_index": ai, "target_index": None, "target_type": "enemy_hero"})

        return orders

    # ================================================================
    #  大模型决策 (LLM)
    # ================================================================
    def _llm_decide(self, game_state: GameState) -> Dict:
        payload = self._build_llm_payload(game_state)
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"调用大模型 ({self.model})...")
                resp = requests.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                decision = self._parse_llm_response(content)
                self._update_memory(game_state, decision)
                self.logger.info(f"大模型决策: {decision.get('reason', '')}")
                return decision
            except requests.exceptions.Timeout:
                self.logger.warning(f"大模型请求超时 (尝试 {attempt+1}/{self.max_retries+1})")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"大模型请求失败: {e} (尝试 {attempt+1}/{self.max_retries+1})")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.warning(f"大模型响应解析失败: {e} (尝试 {attempt+1}/{self.max_retries+1})")
            if attempt < self.max_retries:
                time.sleep(1)

        self.logger.warning("大模型决策失败，回退到规则引擎")
        return self._rule_decide(game_state)

    def _build_llm_payload(self, game_state: GameState) -> Dict:
        state_text = self._state_to_text(game_state)
        messages = [{"role": "system", "content": self.system_prompt}]
        for mem in self.memory:
            messages.append({"role": "user", "content": mem.get("state", "")})
            messages.append({"role": "assistant", "content": json.dumps(mem.get("decision", {}), ensure_ascii=False)})
        messages.append({"role": "user", "content": state_text})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 300,
        }

    def _state_to_text(self, game_state: GameState) -> str:
        lines = ["## 当前游戏状态"]
        lines.append(f"回合: {'我方' if game_state.is_our_turn else '敌方'}")
        lines.append(f"我方英雄血量: {game_state.our_health}" + (f" 护甲: {game_state.our_armor}" if game_state.our_armor else ""))
        lines.append(f"敌方英雄血量: {game_state.opponent_health}" + (f" 护甲: {game_state.opponent_armor}" if game_state.opponent_armor else ""))
        lines.append(f"法力水晶: {game_state.our_mana}/{game_state.total_mana}")

        lines.append(f"\n手牌 ({len(game_state.hand_cards)}张):")
        for i, card in enumerate(game_state.hand_cards):
            playable = " (可出)" if card.is_playable else ""
            name_str = f" {card.name}" if card.name else ""
            lines.append(f"  [{i}] 费用:{card.cost} 类型:{card.card_type}{name_str}{playable}")

        lines.append(f"\n我方随从 ({len(game_state.our_minions)}个):")
        for i, m in enumerate(game_state.our_minions):
            attackable = " (可攻击)" if m.can_attack else ""
            name_str = f" {m.name}" if m.name else ""
            mechanics = []
            if m.has_taunt: mechanics.append("嘲讽")
            if m.has_divine_shield: mechanics.append("圣盾")
            if m.has_stealth: mechanics.append("潜行")
            mech_str = f" [{','.join(mechanics)}]" if mechanics else ""
            lines.append(f"  [{i}] {m.attack}/{m.health}{name_str}{attackable}{mech_str}")

        lines.append(f"\n敌方随从 ({len(game_state.opponent_minions)}个):")
        for i, m in enumerate(game_state.opponent_minions):
            mechanics = []
            if m.has_taunt: mechanics.append("嘲讽")
            if m.has_divine_shield: mechanics.append("圣盾")
            if m.has_stealth: mechanics.append("潜行")
            mech_str = f" [{','.join(mechanics)}]" if mechanics else ""
            lines.append(f"  [{i}] {m.attack}/{m.health}{mech_str}")

        return "\n".join(lines)

    def _parse_llm_response(self, content: str) -> Dict:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return json.loads(content)

    def _update_memory(self, game_state: GameState, decision: Dict):
        self.memory.append({
            "state": self._state_to_text(game_state),
            "decision": decision,
        })
        if len(self.memory) > self.max_memory_len:
            self.memory.pop(0)
