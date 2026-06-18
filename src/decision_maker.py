import json
import os
import time
from typing import Dict, List, Optional

import requests

from src.logger import HsBatLogger
from src.state_recognizer import GameState


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

    def _rule_decide(self, game_state: GameState) -> Dict:
        if not game_state.is_our_turn:
            return {"action": "wait", "reason": "等待我方回合"}

        playable_cards = [c for c in game_state.hand_cards if c.is_playable]
        if playable_cards:
            card = playable_cards[-1]
            self.logger.info(f"规则决策: 出牌 [{card.name}](费用{card.cost})")
            return {
                "action": "play_card",
                "card_index": game_state.hand_cards.index(card),
                "target_index": None,
                "target_type": None,
                "reason": f"规则引擎: 出[{card.name}]",
            }

        attackable = [m for m in game_state.our_minions if m.can_attack]
        if attackable:
            if game_state.opponent_minions:
                target = game_state.opponent_minions[0]
                self.logger.info("规则决策: 随从攻击敌方随从")
                return {
                    "action": "attack",
                    "card_index": None,
                    "target_index": 0,
                    "target_type": "enemy_minion",
                    "reason": "规则引擎: 随从攻击",
                }
            self.logger.info("规则决策: 随从攻击敌方英雄")
            return {
                "action": "attack",
                "card_index": None,
                "target_index": None,
                "target_type": "enemy_hero",
                "reason": "规则引擎: 攻击敌方英雄",
            }

        self.logger.info("规则决策: 结束回合")
        return {"action": "end_turn", "card_index": None, "target_index": None, "target_type": None, "reason": "规则引擎: 无操作可做"}

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
        lines = [f"## 当前游戏状态"]
        lines.append(f"回合: {'我方' if game_state.is_our_turn else '敌方'}")
        lines.append(f"我方英雄血量: {game_state.our_health}")
        lines.append(f"敌方英雄血量: {game_state.opponent_health}")
        lines.append(f"法力水晶: {game_state.our_mana}/{game_state.total_mana}")

        lines.append(f"\n手牌 ({len(game_state.hand_cards)}张):")
        for i, card in enumerate(game_state.hand_cards):
            playable = " (可出)" if card.is_playable else ""
            lines.append(f"  [{i}] 费用:{card.cost} {card.name}{playable}")

        lines.append(f"\n我方随从 ({len(game_state.our_minions)}个):")
        for i, m in enumerate(game_state.our_minions):
            attackable = " (可攻击)" if m.can_attack else ""
            lines.append(f"  [{i}] {m.attack}/{m.health}{attackable}")

        lines.append(f"\n敌方随从 ({len(game_state.opponent_minions)}个):")
        for i, m in enumerate(game_state.opponent_minions):
            lines.append(f"  [{i}] {m.attack}/{m.health}")

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

    def build_attack_plan(self, game_state: GameState) -> List[Dict]:
        orders = []
        for i, minion in enumerate(game_state.our_minions):
            if minion.can_attack:
                if game_state.opponent_minions:
                    for j, enemy in enumerate(game_state.opponent_minions):
                        orders.append({
                            "attacker_index": i,
                            "target_index": j,
                            "target_type": "enemy_minion",
                        })
                    break
                else:
                    orders.append({
                        "attacker_index": i,
                        "target_index": None,
                        "target_type": "enemy_hero",
                    })
        return orders
