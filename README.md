# MedCheck

[![CI](https://github.com/knox1019/MedCheck/actions/workflows/ci.yml/badge.svg)](https://github.com/knox1019/MedCheck/actions/workflows/ci.yml)

体检报告解读与用药禁忌核查 Agent —— 一个可运行、可演示、可现场修改的医疗 RAG + Agent 工程样本。

**当前进度：W1–W10 全部完成 · 167 passed / 8 skipped**（跳过的 8 条为依赖 Ollama 的集成测试；排期见 [路线图](#路线图)）

> ⚠️ **免责声明**
> 本项目为技术学习与技术演示用途，全部输出**不构成任何医疗建议、诊断或用药依据**。
> 请勿将本系统的任何结果用于真实的健康决策。如有健康问题，请咨询执业医师。
> 本项目仅使用公开发布的药学资料与合成数据，**不接入、不存储任何真实患者数据**。

---

## 这是什么

一份体检报告交到普通人手里，最难的往往不是看懂数值，而是不知道哪些异常该立刻就医、哪些只是需要复查，以及正在吃的药和异常指标之间有没有冲突。

MedCheck 想解决的问题只有两个：

1. **把报告读懂**：从 PDF 里结构化抽取检验指标，按「请就诊 / 待复查 / 日常管理」分级，**每条结论都标注原文来源与页码**
2. **把风险查出来**：输入正在服用的药物，检索药品说明书中的禁忌与相互作用，给出带引用的风险提示

技术上，这是一个完整的 RAG + Agent 工程实践：文档解析 → 结构化切分 → 向量检索 + 关键词检索 → 重排序 → 引用溯源 → 工具调用 → 评测闭环 → 容器化部署。

![MedCheck 分层架构](docs/architecture.svg)

> 现场演示流程见 [docs/demo.md](docs/demo.md)（5 分钟脚本 + 高频追问应答）。
> 换一台电脑部署见 [SETUP.md](SETUP.md)（两种方式 + 常见问题速查）。

## 现在能做什么

| 能力                     | 状态        |
| ---------------------- | --------- |
| 服务启动与健康检查（`/health`）   | ✅ 已完成     |
| 配置从环境变量注入              | ✅ 已完成     |
| 结构化日志                  | ✅ 已完成     |
| 单元测试（接口 + 配置）          | ✅ 已完成     |
| **PDF 报告解析与指标结构化**     | ✅ 已完成（见 [能力边界](#能力边界)） |
| 知识库构建（切分 / 向量化 / 入库）   | ✅ 657 药 / 1659 条 / 1024 维 |
| 混合检索 + 重排序 + 引用溯源      | ✅ 重排与溯源完成；混合检索经评测否决 |
| 报告解读生成（带强制引用）          | ✅ 主模型不可用自动降级本地备用 |
| **报告核对（异常指标 × 在服药物）** | ✅ 见 [报告核对](docs/design.md#报告核对把异常指标和在服药物连起来) |
| **用药禁忌核查 Agent（工具调用）** | ✅ 3 个工具，模型自主编排 |
| 评测集与批量评测 Harness     | ✅ 检索 69 + 答案 21 + Agent 18 条，共 15 项指标 |
| Docker 化部署             | ✅ 已完成（镜像 1.01GB） |
| **演示界面（上传→指标→核对）**  | ✅ 单页应用，容器内同样可用 |

**说明**：本仓库按周迭代，README 如实反映当前进度，不提前描述未完成的能力。

## 快速开始

环境要求：Python 3.11+ · [uv](https://docs.astral.sh/uv/) · [Ollama](https://ollama.com/download)（向量化用 bge-m3）

```bash
git clone https://github.com/knox1019/MedCheck.git && cd MedCheck
uv sync                                                           # 装依赖，约 30 秒
cp .env.example .env                                              # 配置模型端点，见 SETUP.md 的三选一
ollama pull bge-m3                                                # 向量模型 1.2GB
uv run python -m scripts.build_index data/corpus/drugs_all.jsonl  # 建库，约 17 分钟，只需一次
uv run uvicorn app.main:app --reload                              # 启动
```

浏览器打开 <http://127.0.0.1:8000/>（自动进入演示界面），接口文档在 <http://127.0.0.1:8000/docs>。

**Docker 方式**：`cd docker && docker compose up --build`。

模型端点怎么选（本地 Ollama / DeepSeek / 本地网关）、换机部署全流程、常见问题速查 → **[SETUP.md](SETUP.md)**。

### 解析一份示例报告

```bash
uv run python -m scripts.parse_report data/samples/report_demo.pdf
```

### 核对你自己的报告和用药

先准备知识库（只需一次），再带上在吃的药：

```bash
uv run python -m scripts.build_index data/corpus/drugs_all.jsonl     # 建库，约 17 分钟
uv run python -m scripts.report_check data/samples/report_demo.pdf --meds 阿托伐他汀,布洛芬
```

不带 `--meds` 时只列异常清单，不调模型、不需要 Ollama。

> ⚠️ 注意用 `-m`（模块方式）而不是直接给文件路径。直接跑 `python scripts/parse_report.py`
> 会因为 `sys.path` 只包含 `scripts/` 而报 `ModuleNotFoundError: No module named 'app'`。

输出节选（共 21 条指标，此处只展示 3 条）：

```json
{
  "report_id": "20261022A0031",
  "exam_date": "2026-10-22",
  "items": [
    {
      "name_raw": "丙氨酸氨基转移酶 ALT",
      "name": null,
      "value": 68.0,
      "unit": "U/L",
      "ref_low": 9.0,
      "ref_high": 50.0,
      "flag": "high",
      "page": 1
    },
    {
      "name_raw": "血红蛋白 HGB",
      "value": 108.0,
      "unit": "g/L",
      "ref_low": 130.0,
      "ref_high": 175.0,
      "flag": "low",
      "page": 1
    },
    {
      "name_raw": "C 反应蛋白 CRP",
      "value": 11.2,
      "unit": "mg/L",
      "ref_low": null,
      "ref_high": 8.0,
      "flag": "high",
      "page": 1
    }
  ]
}
```

> 最后一条演示了单边参考范围 `<8` 的处理：下限为 `null`，上限为 `8.0`。

## 评测结果

| 项目          | 结果                          |
| ----------- | --------------------------- |
| 样本指标总数      | 21 条                        |
| 正确解析        | **21 / 21**                 |
| 异常指标判定      | **11 条，与人工核对 100% 一致**      |
| 其中使用字母标记的    | 7 条（4 条 `H` + 3 条 `L`，非箭头，未依赖报告标记） |
| 单边参考范围的     | 1 条（`<8`）                   |

三套评测（检索 69 条 / 答案 21 条 / Agent 18 条）共 15 项确定性指标，Agent 路径工具选择 18/18、答案评测 21/21。

**三个方案经实测否决并留档**：BM25 混合检索（召回已 100%，无收益）、重排分数拒答闸门（误拒 25%）、细粒度切分（退化 3.7 个百分点）。完整数据与失败分析见 **[docs/evaluation.md](docs/evaluation.md)**。

## 能力边界

明确说明「不做什么」比声称全能更可信。以下为实测结论：

| 场景                | 支持情况                     | 说明                                  |
| ----------------- | ------------------------ | ----------------------------------- |
| 单栏版式检验报告          | ✅ 21 / 21 正确            | 当前按固定模板的列索引解析                       |
| 多栏 / 异构版式         | ⚠️ 产出 `unknown`，不静默出错    | 实测双栏样本产生 63 条（含噪声行）、33 条 `unknown` |
| 扫描件（图片型 PDF）      | ❌ 不支持                    | 需 OCR，不在当前范围                        |
| 定性结果（阴性 / 阳性 / `+`） | ❌ 跳过该行                   | 后续版本处理                              |

**设计原则**：解析不确定时标记为 `unknown`，绝不猜测一个参考范围。宁可少一条结论，也不要给错误结论。

## 演示界面

访问 `http://127.0.0.1:8000/` 会自动跳到 `/ui/`——单页应用，无构建工具，纯原生 JS：

```
上传 PDF → 指标表格（异常高亮，偏高红/偏低蓝）→ 填药 → 核对结果 + 工具轨迹 + 出处
```

四步是渐进展示的：没上传时看不到表格，没填药时看不到核对按钮。演示时一路点下来就是完整故事。

工具调用轨迹会展示出来（`调用 check_indicators({...}) · 返回 570 字`）——**让人看见「Agent 查了什么」，比只给一个黑盒答案有说服力**。

Docker 镜像里也包含界面（`COPY web ./web`），容器起来后同样访问 8000 端口即可。

**界面暴露过一个真 bug**：正文引用 [1]~[7]，出处栏却只有 [1][2]。排查发现根源不在前端——`check_indicators` 给每条证据发独立编号，但「ALT 的禁忌」和「AST 的禁忌」是同一段原文，7 条证据实际只有 2 个不同章节。修法是**在 `ToolContext.register()` 发编号时就按 `chunk_id` 去重**，重复命中只标「同上」不再贴原文。编号必须在产生时就唯一，等展示层去重已经晚了——模型已经把 [6] 写进答案了。

## 技术栈

| 层         | 选型                              | 选择理由                                       |
| --------- | ------------------------------- | ------------------------------------------ |
| 语言 / 运行时  | Python 3.11                     | 生态最完整，医学 NLP 与 LLM 工具链支持最好                 |
| 包管理       | uv                              | 依赖解析快，锁文件可复现                               |
| Web 框架    | FastAPI + Pydantic v2           | 类型驱动、自动生成文档、声明式校验                          |
| 配置管理      | pydantic-settings               | 环境变量与类型校验一体，便于容器化                          |
| PDF 解析    | PyMuPDF（正文）+ pdfplumber（表格）      | 两者强项不同：前者抽正文干净，后者能识别表格结构。实测需配合使用           |
| Embedding | BGE-M3（Ollama 本地部署，1024 维）       | 中文医学语料表现好，可完全离线，契合私有化部署要求。已做 provider 抽象，可切换云端 API |
| 向量库       | Chroma                          | 零运维、单文件持久化，便于私有化部署与演示迁移                    |
| 关键词检索     | ~~rank_bm25~~（经评测否决）          | BM25 混合检索实测无收益（召回已 100%），仅保留向量检索；见评测档案  |
| 重排序       | bge-reranker-v2-m3              | 提升 top-k 精度，可离线                            |
| 大模型       | DeepSeek / Qwen（OpenAI 兼容协议）    | 接口层做了抽象，可平滑切换到本地 vLLM                      |
| Agent     | 原生 Function Calling             | 不依赖框架黑盒，便于排查与讲解                            |
| 部署        | Docker + docker-compose         | 支持离线与内网环境                                  |

**刻意不选**：LangChain（抽象层过厚，出问题难定位）、Streamlit（无法体现前后端接口分离）。

## 目录结构

```
medcheck/
├── app/
│   ├── main.py              # FastAPI 入口，启动链路的组装处
│   ├── api/
│   │   ├── routes.py         # 两个业务接口：解析报告 / 问答
│   │   └── schemas.py        # 请求与响应的数据契约
│   ├── core/
│   │   ├── config.py        # 配置：环境变量 > .env > 默认值
│   │   └── logging.py       # 结构化日志配置
│   ├── ingest/
│   │   ├── models.py        # LabItem / Report 数据结构
│   │   ├── indicators.py    # 指标 → 说明书写法的映射（连接报告与语料）
│   │   └── pdf_parser.py    # PDF → 结构化指标（5 个函数分层）
│   ├── report/
│   │   ├── analyzer.py      # 报告核对：异常指标 × 在服药物章节（关键词召回）
│   │   └── narrative.py     # 报告解读：让模型判断依据与指标的真实关系
│   ├── rag/
│   │   ├── embedder.py      # 向量化：分批、指数退避重试、L2 归一化
│   │   ├── store.py         # Chroma 读写：重建集合、向量检索、元数据过滤
│   │   ├── guards.py        # 实体门控：从问题里识别药名，区分「库内 / 库外」
│   │   ├── reranker.py      # cross-encoder 重排（ONNX int8，懒加载 + 单例）
│   │   ├── citation.py      # 出处模型：说明书按章节、报告按页码
│   │   └── retriever.py     # 检索编排：门控 → 过滤 → 阈值 → 重排 → 出处
│   ├── agent/
│   │   ├── tools.py          # 3 个工具：查说明书 / 核对指标 / 列可查药品
│   │   └── loop.py           # 编排循环：模型决策 → 执行工具 → 汇总答案
│   └── llm/
│       ├── client.py        # OpenAI 兼容对话客户端（重试 + 温度 0）
│       ├── prompts.py       # 提示词、资料编号、拒答短语契约
│       └── qa.py            # 问答编排：检索 → 上下文 → 模型 → 解析
├── eval/
│   ├── agentset.jsonl       # Agent 评测集 18 条（标注期望工具）
│   ├── answerset.jsonl      # 答案级评测集 21 条（含关键点标注）
│   ├── report_agent.md      # Agent 评测报告（自动生成）
│   ├── report_answer.md     # 答案级评测报告（自动生成）
│   ├── goldset.jsonl        # 基础评测集 42 条
│   ├── goldset_hard.jsonl   # 困难评测集 27 条（口语化提问）
│   ├── report.md            # 基础集评测报告（自动生成）
│   └── report_hard.md       # 困难集评测报告（自动生成）
├── scripts/
│   ├── probe_pdf.py         # 探针：观察 PDF 真实结构（写解析器前必跑）
│   ├── parse_report.py      # 命令行入口：解析一份报告
│   ├── report_check.py      # 命令行入口：报告核对（--meds 指定在服药物）
│   ├── agent_chat.py        # 命令行入口：Agent 模式（模型自主调用工具）
│   ├── build_index.py       # 建库：语料 → 向量化 → 写入 Chroma
│   ├── build_drug_index.py  # 由国家基本药物目录生成药名索引
│   ├── scrape_labels.py     # 抓取说明书章节（串行 + 延迟，可续跑）
│   ├── merge_corpus.py      # 合并人工核对版与抓取版语料
│   ├── split_corpus.py      # 细粒度切分：按分号切 + 贪婪合并（实验用，默认未启用）
│   ├── search.py            # 检索命令行，支持 --drug 元数据过滤
│   ├── ask.py               # 问答命令行：带引用的答案
│   ├── probe_llm.py         # 模型探活：端点/鉴权/思考过程/token 占比
│   ├── compare_llm.py       # 模型对比：同一批问题跑多个模型，比较引用率与延迟
│   ├── try_rerank.py        # 学习用：对比重排前后的排序差异
│   ├── run_eval.py          # 检索评测：跑 goldset，输出指标与报告
│   └── run_answer_eval.py   # 答案评测：五项确定性指标 + 药名幻觉检测
├── data/
│   ├── corpus/
│   │   ├── drugs.jsonl          # 语料：28 药 85 条说明书章节
│   │   ├── drugs_split.jsonl    # 切分实验产物（未启用）
│   │   ├── drug_name_index.json # 官方药名索引 663 条
│   │   ├── drugs_web.jsonl      # 抓取的说明书章节（651 药 / 1953 条）
│   │   └── drugs_all.jsonl      # 合并语料（657 药 / 1659 条，建库用）
│   ├── samples/             # 合成样本报告（无真实患者数据）
│   └── chroma/              # 向量库（可重建，不入库）
├── web/                     # 演示用单页界面
├── docker/                  # 镜像与编排
├── docs/                    # 深挖文档（设计决策 / 评测档案 / 踩坑 / 演示脚本 / 架构图）
├── eval/                    # 三套评测集与自动生成的报告
├── SETUP.md                 # 换机部署指南
└── tests/                   # 单元测试（167 个通过，8 个需 Ollama）
```

## 路线图

| 周次  | 目标      | 交付物                      | 状态 |
| --- | ------- | ------------------------ | -- |
| W1  | 项目骨架    | 服务可启动，`/health` 可用，容器可构建 | ✅ 全部完成（镜像 1.01GB） |
| W2  | PDF 解析  | 报告转结构化 JSON              | ✅ 21/21 正确 |
| W3  | 知识库     | 一条命令建库，检索可用              | ✅ 657 药 / 1659 条 / 1024 维 |
| W4  | 检索增强    | 混合检索 + 重排 + 引用溯源         | ✅ 重排 92.6%、溯源完成；BM25 经评测否决 |
| W5  | 生成层     | 端到端问答，强制引用，检索不到即拒答       | ✅ 问答 + 答案评测 + 报告核对均完成 |
| W6  | Agent 化 | 多工具自主编排                  | ✅ 3 工具 + 轮数保护 + 全局引用编号 |
| W7  | 评测体系    | 评测集 + 批量跑分报告             | ✅ 检索/答案/Agent 三套，共 15 项指标 |
| W8  | 部署      | 离线一键部署，排障手册              | ✅ Docker 实测通过，见 SETUP.md |
| W9  | 演示      | 界面 + 架构图 + 演示脚本          | ✅ 全部完成 |
| W10 | 包装      | 完整文档与面试材料                | ✅ README + docs/ 四份文档 + 演示脚本 |

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [SETUP.md](SETUP.md) | 换机部署两种方式、常见问题速查 |
| [docs/design.md](docs/design.md) | 设计决策：三层拒答、词表映射、提示词迭代、Agent 编排、接口设计、容器细节 |
| [docs/evaluation.md](docs/evaluation.md) | 三套评测完整数据与负结果档案 |
| [docs/lessons.md](docs/lessons.md) | 踩坑记录 40+ 条，按模块分类 |
| [docs/demo.md](docs/demo.md) | 5 分钟现场演示脚本与高频追问应答 |
| [docs/architecture.svg](docs/architecture.svg) | 分层架构图 |

## 合规设计

医疗场景的数据合规不是可选项，本项目在设计阶段即作出以下约束：

- **数据来源**：仅使用公开发布的药品说明书、诊疗指南、专家共识
- **数据边界**：不接入任何真实患者病历、院内系统或脱敏数据；演示用报告全部为 WPS 手工合成的假数据（姓名统一为「张三」）
- **输出边界**：只做信息检索与分级提示，不输出诊断结论，不给用药剂量建议
- **可追溯**：`LabItem` 从解析阶段即记录来源页码（`page` 字段），为后续引用溯源预留，避免后期返工
- **界面约束**：展示页固定免责声明

## License

MIT
