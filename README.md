# HsBat - 炉石传说混合架构自动化脚本

## 设计理念

纯大模型方案延迟较高（每次 1-5 秒），不适合炉石传说这种回合有时间限制的游戏。采用 **传统 CV + 大模型** 混合架构：

- **底层 (经典 CV)**：OpenCV 模板匹配 + Tesseract OCR 快速识别固定 UI 元素（按钮、法力水晶、血量）
- **高层 (大模型)**：仅在需要策略决策时调用 LLM API（如选择出哪张牌、攻击哪个目标）

## 项目结构

```
HsBat/
├── main.py                  # 主入口
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖
├── README.md
├── templates/               # 模板图片目录（用户自行截取）
│   ├── end_turn.png         # 结束回合按钮
│   └── attack.png           # 攻击按钮
├── src/
│   ├── __init__.py
│   ├── state_recognizer.py  # 模块1: 快速状态识别 (OpenCV + OCR)
│   ├── decision_maker.py    # 模块2: 策略决策 (规则引擎 + 大模型 API)
│   ├── action_executor.py   # 模块3: 动作执行 (PyAutoGUI + 贝塞尔曲线)
│   ├── game_controller.py   # 主控制器 - 游戏循环调度
│   ├── debug_ui.py          # 调试可视化 (OpenCV 窗口)
│   └── logger.py            # 日志模块
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
| `end_turn.png` | 结束回合按钮截图 | 截取完整的按钮区域 |
| `attack.png` | 攻击按钮截图 | 截取完整的按钮区域 |

### 4. 配置大模型 API (可选)

```bash
# 方式1: 环境变量
set HSBAT_LLM_API_KEY=your_api_key_here

# 方式2: 修改 config.yaml 中的 api_key
```

支持任何 OpenAI 兼容的 API (OpenAI、Azure、Claude、本地 LLM等)。

## 使用方法

### 基础运行 (纯规则引擎)

```bash
python main.py --rule-only
```

### 带大模型决策

```bash
python main.py
```

### 调试模式

```bash
python main.py --debug
```

### 仅识别不执行 (安全试运行)

```bash
python main.py --dry-run
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `-c, --config PATH` | 配置文件路径 (默认: config.yaml) |
| `-d, --debug` | 启用调试模式 |
| `--no-debug-ui` | 禁用调试可视化窗口 |
| `--rule-only` | 仅使用规则引擎，不调用大模型 |
| `--dry-run` | 仅运行识别模式，不执行任何鼠标操作 |

## 架构详解

### 模块 1: 状态识别 (StateRecognizer)

- **模板匹配**: 使用 `cv2.matchTemplate` 识别按钮
- **法力水晶识别**: 颜色阈值 + 轮廓检测
- **血量识别**: OCR 数字识别
- **手牌检测**: Canny 边缘检测 + 轮廓分析
- **随从检测**: 区域边缘检测 + OCR 攻防数值
- **回合判断**: OC视觉识别回合文字

### 模块 2: 策略决策 (DecisionMaker)

- **规则引擎** (快速，无需网络):
  - 优先出牌（从高费到低费）
  - 随从优先攻击敌方随从
  - 无操作可做则结束回合

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

### 调试可视化窗口

- 显示实时游戏画面 + 识别标注
- 快捷键:
  - `q`: 关闭调试窗口
  - `s`: 保存当前调试截图
  - `Space`: 暂停/继续

## 配置说明

主要配置项在 `config.yaml`:

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `screen.game_region` | 游戏窗口区域 | 1920x1080 |
| `templates.match_threshold` | 模板匹配阈值 | 0.75 |
| `llm.enabled` | 启用大模型 | true |
| `llm.api_base` | API 地址 | https://api.openai.com/v1 |
| `llm.model` | 模型名称 | gpt-4o |
| `action.bezier_points` | 贝塞尔曲线点数 | 20 |
| `debug.show_debug_window` | 显示调试窗口 | true |

## 安全提示

- 运行前请确保炉石传说窗口在正确位置
- Failsafe: 鼠标移到屏幕左上角 (0,0) 可立即停止
- 建议先使用 `--dry-run` 测试识别效果
- 不要在排位赛等重要对局中首次使用

## 输出示例

```
2024-01-15 14:30:01 [INFO] Main - HsBat v1.0 启动
2024-01-15 14:30:01 [INFO] Main - 大模型决策: 开启
2024-01-15 14:30:02 [DEBUG] StateRecognizer - 状态: 回合=敌方 血量=30/30 法力=0/0 手牌=0张
2024-01-15 14:30:03 [INFO] GameController - 检测到我方回合开始
2024-01-15 14:30:03 [INFO] GameController - === 第 1 回合 ===
2024-01-15 14:30:04 [INFO] DecisionMaker - 调用大模型 (gpt-4o)...
2024-01-15 14:30:06 [INFO] DecisionMaker - 大模型决策: 出[精灵弓箭手](费用1)
2024-01-15 14:30:06 [INFO] GameController - 执行动作: play_card - 大模型: 出[精灵弓箭手]
```

## 许可证

MIT License
