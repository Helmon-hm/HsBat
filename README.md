# HsBat - 炉石传说混合架构自动化脚本

## 设计理念

纯大模型方案延迟较高（每次 1-5 秒），不适合炉石传说这种回合有时间限制的游戏。采用 **传统 CV + 大模型** 混合架构：

- **底层 (经典 CV)**：OpenCV 模板匹配 + Tesseract OCR 快速识别固定 UI 元素（按钮、法力水晶、血量）
- **高层 (大模型/规则引擎)**：策略决策，大模型智能决策或规则引擎快速响应，两种模式可在 GUI 中一键切换

## 项目结构

```
HsBat/
├── main.py                  # 主入口
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖
├── README.md
├── templates/               # 模板图片目录（用户自行截取）
│   └── end_turn.png         # 结束回合按钮（可选，提供则用模板匹配，否则走 OCR）
├── src/
│   ├── __init__.py
│   ├── state_recognizer.py  # 模块1: 快速状态识别 (OpenCV + OCR)
│   ├── decision_maker.py    # 模块2: 策略决策 (规则引擎 + 大模型 API)
│   ├── action_executor.py   # 模块3: 动作执行 (PyAutoGUI + 贝塞尔曲线)
│   ├── game_controller.py   # 主控制器 - 游戏循环调度
│   ├── gui_app.py           # GUI 界面 (tkinter)
│   ├── logger.py            # 日志模块
│   └── paths.py             # 路径工具
├── screenshots/             # 调试截图输出目录
└── logs/                    # 日志输出目录
```

## 系统要求

- Python 3.9+
- Windows 10/11 (推荐，截取自 Windows 炉石客户端)
- Tesseract-OCR (用于卡牌文字识别)
- 炉石传说客户端

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Tesseract-OCR

- 下载地址: https://github.com/UB-Mannheim/tesseract/wiki
- 安装时勾选简体中文语言包
- 默认路径: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- 安装后在 `config.yaml` 中配置 `ocr.tesseract_path`

### 3. 准备模板图片

在 `templates/` 目录下放入截取的模板图片:

| 文件名 | 说明 | 建议 |
|--------|------|------|
| `end_turn.png` | 结束回合按钮截图 | 截取右下角蓝色"结束回合"按钮区域 |

> **注意**：炉石传说中没有独立的"攻击按钮"，可攻击的随从通过**绿色边框光晕**来表示。程序通过 HSV 颜色检测自动识别绿色边框，无需额外模板。

### 4. 配置大模型 API (可选)

```bash
# 方式1: 环境变量
set HSBAT_LLM_API_KEY=your_api_key_here

# 方式2: 修改 config.yaml 中的 api_key
```

支持任何 OpenAI 兼容的 API (OpenAI、Azure、Claude、本地 LLM等)。

## 使用方法

### GUI 模式 (默认)

```bash
python main.py
```

启动图形界面，可在设置页中配置所有参数，点击"启动"开始 Bot。

### 命令行模式

```bash
python main.py --cli
```

### 规则引擎快速模式

```bash
python main.py --cli --rule-only
```

### 仅识别不执行 (安全试运行)

```bash
python main.py --cli --dry-run
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `-c, --config PATH` | 配置文件路径 (默认: config.yaml) |
| `--cli` | 使用命令行模式 (默认启动 GUI) |
| `--rule-only` | 仅使用规则引擎，不调用大模型 |
| `--dry-run` | 仅运行识别模式，不执行任何鼠标操作 |

## 架构详解

### 模块 1: 状态识别 (StateRecognizer)

- **屏幕分辨率自动检测**: 启动时通过 `pyautogui.size()` 获取实际分辨率，所有识别区域使用相对百分比自适应
- **模板匹配**: 使用 `cv2.matchTemplate` 识别结束回合按钮
- **绿色边框检测**: HSV 颜色空间检测可攻击随从的绿色边框光晕
- **法力水晶识别**: 颜色阈值 + 轮廓检测
- **血量识别**: OCR 数字识别
- **手牌检测**: Canny 边缘检测 + 轮廓分析
- **随从检测**: 区域边缘检测 + OCR 攻防数值
- **回合判断**: OCR 识别回合文字（无 Tesseract 时通过按钮隐藏判断）
- **对局结束检测**: OCR 识别"胜利/失败/Victory/Defeat"等关键词
- **主菜单检测**: OCR 识别"开始/对战/Play"等按钮文字

### 模块 2: 策略决策 (DecisionMaker)

- **规则引擎** (快速，无需网络):
  - **出牌策略**: 优先出高费 / 优先出低费（铺场）
  - **攻击策略**: 智能（斩杀优先、危险解场） / 只打脸 / 只解场
  - **危险防御**: 敌方场攻超过斩杀线时自动切换解场模式
  - **斩杀检测**: 场攻 ≥ 敌方血量时直接打脸斩杀
  - **效率交换**: 贪心算法找最优随从交换对
  - **英雄技能**: 剩余费用 ≥ 2 时自动使用

- **大模型引擎** (智能，需 API):
  - 将结构化游戏状态转为文本提示
  - 调用 LLM API 获取决策
  - 记忆最近 5 轮状态做上下文
  - 失败时自动回退到规则引擎

### 模块 3: 动作执行 (ActionExecutor)

- **贝塞尔曲线鼠标轨迹**: 模拟自然鼠标移动路径，避免被检测
- **随机延迟**: 每步操作加入 150-700ms 随机延迟
- **拖拽出牌**: 模拟从手牌拖拽到战场的操作
- **Failsafe**: 鼠标移到左上角可紧急停止

### 对局自动重排队

对局结束后自动：
1. 检测到游戏结束画面
2. 反复点击跳过结算/奖励动画
3. 回到主菜单后检测并点击"开始"按钮
4. 进入新对局继续 Bot

可在 GUI 设置页的"对局结束后自动开始下一局"开关控制。

### 调试截图

启用 `debug.save_screenshots: true` 后，每次状态识别会覆盖写入 `screenshots/latest/` 目录：
- `00_full.png` — 完整截图
- `mana_region.png` / `health_region.png` 等 — 各识别区域裁剪
- `card_XX_牌名.png` — 每张手牌裁剪
- `our_minion_XX_atk.png` / `opp_minion_XX.png` — 每个随从裁剪

同时会在 `screenshots/` 下存一份 `latest_full.png`，每次覆盖。

## 配置说明

主要配置项在 `config.yaml`:

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `screen.game_region.width/height` | 游戏窗口区域 | 0=自动检测 |
| `game.*_region` | 各识别区域百分比 | 百分比值 |
| `templates.match_threshold` | 模板匹配阈值 | 0.75 |
| `llm.enabled` | 启用大模型 | true |
| `llm.api_base` | API 地址 | https://api.openai.com/v1 |
| `llm.model` | 模型名称 | gpt-4o |
| `rule_engine.play_card_strategy` | 出牌策略 | high_cost_first |
| `rule_engine.attack_strategy` | 攻击策略 | smart |
| `game.auto_requeue` | 对局结束自动开始下一局 | true |
| `action.bezier_points` | 贝塞尔曲线点数 | 20 |
| `debug.save_screenshots` | 保存调试区域截图 | true |

## 安全提示

- 运行前请确保炉石传说窗口在正确位置
- Failsafe: 鼠标移到屏幕左上角 (0,0) 可立即停止
- 建议先使用 `--dry-run` 测试识别效果
- 不要在排位赛等重要对局中首次使用

## 输出示例

```
2026-06-18 16:01:06 [INFO] Main - HsBat v2.0 GUI 模式启动
2026-06-18 16:01:06 [INFO] Main - 决策模式: 规则引擎
2026-06-18 16:01:06 [INFO] StateRecognizer - 自动检测屏幕分辨率: 3840 x 2160
2026-06-18 16:01:07 [DEBUG] StateRecognizer - 状态: 回合=我方 血量=30/30 法力=5/5 手牌=3张
2026-06-18 16:01:07 [INFO] GameController - === 第 1 回合 ===
2026-06-18 16:01:07 [INFO] DecisionMaker - 规则决策: 出牌 [精灵弓箭手](费用1, 策略=low_cost_first)
2026-06-18 16:01:07 [INFO] GameController - 执行动作: play_card - 规则引擎: 出牌 [精灵弓箭手]
```

## 许可证

MIT License
