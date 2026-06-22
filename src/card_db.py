import os
import re
from typing import Dict, Optional, Tuple


class CardDatabase:
    SIMPLE_DB = {
        "HERO_01": "加尔鲁什·地狱咆哮",
        "HERO_02": "萨尔",
        "HERO_03": "瓦莉拉·萨古纳尔",
        "HERO_04": "乌瑟尔·光明使者",
        "HERO_05": "雷克萨",
        "HERO_06": "玛法里奥·怒风",
        "HERO_07": "古尔丹",
        "HERO_08": "吉安娜·普罗德摩尔",
        "HERO_09": "安度因·乌瑞恩",
        "HERO_10": "伊利丹·怒风",
        "HERO_11": "死亡骑士",
        "HERO_12": "恶魔猎手",
        "CS1h_001": "次级治疗术",
        "CS2_017": "变形",
        "CS2_022": "冰锥术",
        "CS2_023": "奥术智慧",
        "CS2_024": "冰霜新星",
        "CS2_025": "奥术飞弹",
        "CS2_026": "冰霜震击",
        "CS2_027": "镜像",
        "CS2_028": "暴风雪",
        "CS2_029": "火球术",
        "CS2_031": "冰枪术",
        "CS2_032": "烈焰风暴",
        "CS2_033": "水元素",
        "CS2_034": "火冲",
        "CS2_037": "闪电链",
        "CS2_038": "先祖之魂",
        "CS2_039": "风怒",
        "CS2_041": "治疗波",
        "CS2_042": "火舌图腾",
        "CS2_045": "石化武器",
        "CS2_046": "嗜血",
        "CS2_049": "图腾之力",
        "CS2_050": "灼热图腾",
        "CS2_051": "石爪图腾",
        "CS2_052": "空气之怒图腾",
        "CS2_053": "远视",
        "CS2_056": "灵魂之火",
        "CS2_057": "暗影箭",
        "CS2_059": "血之小鬼",
        "CS2_061": "吸取生命",
        "CS2_062": "地狱烈焰",
        "CS2_063": "腐蚀术",
        "CS2_064": "恐惧地狱火",
        "CS2_065": "虚空行者",
        "CS2_072": "背刺",
        "CS2_073": "冷血",
        "CS2_074": "致命药膏",
        "CS2_075": "影袭",
        "CS2_076": "刺杀",
        "CS2_077": "疾跑",
        "CS2_080": "刺骨",
        "CS2_082": "邪恶短剑",
        "CS2_083": "匕首精通",
        "CS2_084": "猎人印记",
        "CS2_087": "力量祝福",
        "CS2_088": "王者祝福",
        "CS2_089": "圣光术",
        "CS2_091": "保护之手",
        "CS2_092": "奉献",
        "CS2_093": "生而平等",
        "CS2_094": "愤怒之锤",
        "CS2_097": "真银圣剑",
        "CS2_101": "援军",
        "CS2_102": "铜墙铁壁",
        "CS2_103": "冲锋",
        "CS2_104": "狂暴",
        "CS2_105": "英勇打击",
        "CS2_106": "斩杀",
        "CS2_108": "顺劈斩",
        "CS2_114": "旋风斩",
        "CS2_118": "奥金斧",
        "CS2_119": "淡水鳄",
        "CS2_120": "血沼迅猛龙",
        "CS2_121": "霜狼步兵",
        "CS2_122": "团队领袖",
        "CS2_124": "狼骑兵",
        "CS2_125": "铁鬃灰熊",
        "CS2_127": "银背族长",
        "CS2_131": "暴风城勇士",
        "CS2_141": "阿拉希武器匠",
        "CS2_142": "库卡隆精英卫士",
        "CS2_146": "南海船工",
        "CS2_147": "侏儒发明家",
        "CS2_150": "侏儒列兵",
        "CS2_151": "白银之手骑士",
        "CS2_152": "大检察官",
        "CS2_155": "大法师",
        "CS2_161": "拉文霍德刺客",
        "CS2_162": "霜狼督军",
        "CS2_168": "鱼人袭击者",
        "CS2_169": "幼龙鹰",
        "CS2_171": "石牙野猪",
        "CS2_172": "血沼迅猛龙",
        "CS2_173": "淡水鳄",
        "CS2_179": "森金持盾卫士",
        "CS2_182": "冰风雪人",
        "CS2_186": "作战傀儡",
        "CS2_187": "石拳食人魔",
        "CS2_188": "鲁莽火箭兵",
        "CS2_189": "精灵弓箭手",
        "CS2_196": "沼泽爬行者",
        "CS2_197": "食人魔法师",
        "CS2_200": "藏宝海湾保镖",
        "CS2_201": "熔核猎犬",
        "CS2_213": "残阳祭司",
        "CS2_222": "奥数傀儡",
        "CS2_226": "狼人渗透者",
        "CS2_227": "风险投资公司雇佣兵",
        "CS2_231": "风语者",
        "CS2_232": "火元素",
        "CS2_233": "土元素",
        "CS2_234": "暗影形态",
        "CS2_235": "北郡牧师",
        "CS2_236": "神圣新星",
        "CS2_237": "心灵视界",
    }

    def __init__(self):
        self._db = dict(self.SIMPLE_DB)
        self._load_hearthstone_data()

    def _load_hearthstone_data(self):
        try:
            from hearthstone import cardxml
            import hearthstone_data

            db_path = None
            if hasattr(hearthstone_data, "get_cardxml_path"):
                db_path = hearthstone_data.get_cardxml_path()
            elif hasattr(hearthstone_data, "__path__"):
                for p in hearthstone_data.__path__:
                    candidate = os.path.join(p, "CardDefs.xml")
                    if os.path.exists(candidate):
                        db_path = candidate
                        break

            if db_path and os.path.exists(db_path):
                cards, _ = cardxml.load(db_path)
                for card in cards:
                    card_id = getattr(card, "CardID", "") or getattr(card, "id", "")
                    name = getattr(card, "name", "") or getattr(card, "Name", "")
                    if card_id and name:
                        self._db[card_id] = name
                import logging
                logging.getLogger("HsBat.CardDatabase").info(f"从hearthstone_data加载了 {len(cards)} 张卡牌")
            else:
                import logging
                logging.getLogger("HsBat.CardDatabase").info("hearthstone_data 未安装或 CardDefs.xml 不存在，使用内置卡牌数据库")
        except ImportError:
            import logging
            logging.getLogger("HsBat.CardDatabase").info("hearthstone_data 未安装，使用内置卡牌数据库")
        except Exception as e:
            import logging
            logging.getLogger("HsBat.CardDatabase").warning(f"加载hearthstone_data失败: {e}，使用内置卡牌数据库")

    def get_name(self, card_id: str) -> str:
        if not card_id:
            return "未知卡牌"
        name = self._db.get(card_id, "")
        if name:
            return name
        if card_id.startswith("HERO_"):
            return "英雄"
        if card_id.startswith("CS"):
            return f"卡牌({card_id})"
        match = re.match(r".*_(\d+)$", card_id)
        if match:
            return f"卡牌({card_id})"
        return card_id

    def get_type(self, card_id: str) -> str:
        if not card_id:
            return "unknown"
        if card_id.startswith("HERO_") or card_id.endswith("_HERO"):
            return "hero"
        if card_id.endswith("_HP") or card_id.endswith("_HERO_POWER"):
            return "hero_power"
        return "unknown"

    def get_card_id(self, card_id: str) -> str:
        return card_id


_card_db_instance: Optional[CardDatabase] = None


def get_card_db() -> CardDatabase:
    global _card_db_instance
    if _card_db_instance is None:
        _card_db_instance = CardDatabase()
    return _card_db_instance
