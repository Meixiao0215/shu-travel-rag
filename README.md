# 差旅报销 RAG Agent  面向高校报销场景的智能问答助手

从空目录亲手搭建请阅读 [从零构建到发布 GitHub 的完整教程](docs/from_zero_to_github.md)，其中包含逐步说明、完整文件代码、运行命令和验收任务。

一个适合 LangChain 入门后亲手理解、修改和复现的项目：从学校 PDF 中检索报销规则，返回带条款与页码的回答，并在条件或证据不足时追问或说明限制。

聊天模型：`deepseek-v4-flash`。本地向量模型：`BAAI/bge-m3`。数据源为用户提供的上大内〔2026〕128号文件。本项目使用 LangChain 基础组件，解析、表格还原、检索融合、证据管理、评测和页面代码均在本仓库中。

## 先运行，再按模块学习

当前电脑已建好 `.venv`、解析数据与向量索引。进入项目目录：

```bash
cd "/Users/shoremay/Documents/ChatGPT/langchain agent"
source .venv/bin/activate
python -m travel_rag.cli doctor
```

先用完全不需要 API 密钥的命令验证检索：

```bash
python -m travel_rag.cli search "二级教授乘高铁可以选什么座位？" --mode bm25
python -m travel_rag.cli search "横向项目自驾，汽油费能报吗？" --mode hybrid
```

打开本地 `.env`，只在 `DEEPSEEK_API_KEY=` 后填写你自己的密钥。不需要把密钥发到对话中。其他配置已经写好。如果页面服务已经启动，填写后先在对应终端按 Ctrl+C 停止，再重新运行启动命令。

```bash
python -m travel_rag.cli ask "横向科研项目自驾，汽油费和通行费能报销吗？"
python -m travel_rag.cli chat
```

`chat` 中输入 `/quit` 退出。可以先问会议统一承担食宿的补助，再追问“那食宿自理呢”。

运行聊天页面：

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

在浏览器访问终端给出的本地地址。没有密钥时，侧栏选择“仅搜索原文”与“关键词检索”，依然可以检查条款。

## 在新电脑从头搭建

推荐 Python 3.12。以下目录名只是示例，进入自己创建的项目目录操作。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
cp .env.example .env
```

`requirements.lock` 固定本次实测依赖版本；`pyproject.toml` 说明直接依赖范围。也可以用 `python -m pip install -e ".[dev]"` 重新解析依赖，但新组合需要重新验证。

把本次学校 PDF 放在本地，执行：

```bash
python -m travel_rag.cli ingest "/完整路径/上大内〔2026〕128号-关于印发《上海大学科研国内差旅费管理办法》的通知.pdf"
python -m travel_rag.cli index
```

本机已缓存 BGE-M3，所以 `.env` 默认 `EMBEDDING_LOCAL_ONLY=true`。新电脑若没有模型缓存，首次联网下载时改为 `false`，建好索引后可改回 `true`。BGE-M3 权重较大；需要更轻的中文模型时可改为 `BAAI/bge-small-zh-v1.5`，重新建立索引并比较效果。聊天模型 API 不负责生成这些向量。

数据和索引都是本地生成。PDF、`.env`、缓存、索引和运行日志已加入 `.gitignore`；分享代码时由使用者自行导入文件。

## 结构与调用过程

```text
PDF
 └─ ingest.py：按坐标还原两张表 → 合并跨页条款 → 小块与父条款
     ├─ pages.json：原页文本与表格整理文本
     ├─ articles.json：完整条款、来源、页码
     └─ chunks.json：用于检索的小块，指向父条款

用户问题 + 最近对话
 └─ retrieval.py：BM25 / Chroma 向量 / RRF 混合检索
     └─ 回收完整父条款 + 适用范围证据
         └─ agent.py：DeepSeek + create_agent
             ├─ search_policy：补查条款，最多3次
             ├─ read_policy_page：核对原页，最多2次
             └─ schemas.py：结构化结论 → 引用校验 → 页面展示
```

第一次检索由程序执行，保证每次政策问答都有初始证据；Agent 决定是否补查。普通 RAG 对照使用相同初始检索和回答结构，但只生成一次，不提供工具循环。

```bash
python -m travel_rag.cli ask "举办方统一安排且承担食宿，补助怎么算？" --baseline
python -m travel_rag.cli ask "举办方统一安排且承担食宿，补助怎么算？"
```

## 你应该按这个顺序读代码

| 顺序 | 文件 | 学习目标 | 自己动手的验收任务 |
|---|---|---|---|
| 1 | `travel_rag/ingest.py` | PDF文字与表格、元数据、父子块 | 找到第七条，解释为何横跨第3、4页；核对二级教授火车标准 |
| 2 | `travel_rag/retrieval.py` | Embeddings接口、持久化、BM25、RRF | 打印同一问题三种检索的前5条；解释差异 |
| 3 | `travel_rag/schemas.py` | Pydantic与证据协议 | 传入假的证据编号，确认程序拒绝展示 |
| 4 | `travel_rag/agent.py` | Model、Messages、Tool Calling、create_agent、LCEL | 画出一次补查的用户→模型→工具→模型流程 |
| 5 | `travel_rag/evaluate.py` | 从样例到可复现评测 | 分析一个未找全证据的问题，先定位解析还是检索问题 |
| 6 | `app.py`、`travel_rag/cli.py` | 多轮历史、过程流、异常反馈 | 在页面重现一次原文搜索与两轮追问 |
| 7 | `tests/test_pipeline.py` | 防回归测试、模型替身 | 理解为什么替身测试通过不能证明真实模型答对 |

详见 `docs/learning_path.md`。建议先读函数再运行命令，之后关闭参考实现，独立重写一遍 BM25 与 RRF 部分。

## 这份 PDF 的具体处理

- 共12页。正文33条；第1页是通知，第11、12页是附表。
- 第3页交通标准表中，“二级及以上教授”的火车栏与上一行合并。按单元格坐标判断共享关系，不能把空白当作没有标准。
- 第4页住宿标准表有两层表头。转成“省市；人员类别：金额”的逐行记录，保留元/人天单位。
- 第七、十八、二十二、二十七、三十条等跨页。按条文边界合并，再记录完整页码列表。
- 小块约450字符、重叠80字符，是检索起点；传给模型的内容是完整父条款。重复导入保持稳定 ID；重复建索引只补缺失块。
- 对应文档指纹记录在 `data/policy_profile.json`。当前解析器有意只接受这份已检查文件；增加新文件前要重新验证版面和政策版本。
- 通知落款日期为2026年7月28日，办公室印发日期为7月30日；第三十三条写的是“自发文之日起施行”。程序保留这一区别，边界日期不擅自合并解释。

第12条的常规限额不能覆盖所有情况。例如第23条对会议统一安排宾馆作出专门规定，第24条区分横向和纵向项目自驾费用。检索评测包含这类条件差异。

## 评测与测试

```bash
python -m pytest -q
python -m travel_rag.cli eval --mode bm25
python -m travel_rag.cli eval --mode vector
python -m travel_rag.cli eval --mode hybrid
```

`eval/cases.json` 有40道人工按原文设计的种子题：30道开发题、10道留出题。默认只跑开发集；确认设计后再执行 `--split test`。后续应自己增加真实用户问法，并由另一人核对标注。

检索报告保存在 `eval/reports/`：

- `mean_evidence_recall`：每道可回答单轮题所需条款被召回的比例，再求平均。
- `all_evidence_rate`：所有所需条款都出现在前5条的题目比例。
- `mrr`：第一个所需条款的倒数排名均值。

追问、无答案及多轮样例不混入这些单轮检索指标。指标基于人工标注的必要条款，不是所有可能有效证据的穷举。

配置密钥后，可以明确调用模型导出待人工评阅的结果：

```bash
python -m travel_rag.cli eval --generate --limit 5
python -m travel_rag.cli eval --generate --limit 5 --baseline
```

检查报告中的 `review`：结论是否被原文支持、条件是否完整、追问是否必要、无答案时是否克制。不要把“编号存在”当成“结论正确”，也不要把单份文件上的小样本指标宣传为通用准确率。

## 工程细节与实际边界

- DeepSeek 使用 `ChatDeepSeek(model="deepseek-v4-flash")`，显式关闭默认思考模式，超时60秒、SDK重试最多2次。没有改用其他模型。
- `ToolStrategy(PolicyAnswer)` 用工具调用承载结构化回答；普通 RAG 对照使用 JSON mode。
- 工具调用有独立预算，Graph 另有16步上限。结构化输出持续失败或引用不合法时，本次请求失败，不展示未经校验的结果。
- 页面使用**过程流式展示**：检索、补查、完成。最终结论通过校验后一次展示；尚未实现逐 token 展示或原生异步服务，作为后续练习。
- 每次请求创建独立证据池。会话历史由页面或 CLI 持有，重启不持久化；只保留最近6条消息，每条最多4000字符。
- 本地统计日志只存耗时、模型、token数量、调用次数，不存密钥和文档正文。API失败会显示失败类型；真实模型质量仍需配置密钥后验证。
- 检索排序分数不是置信度，不使用未经标定的固定阈值断言“没有答案”。证据不足主要依赖提示词与结果评阅，不能承诺完全消除幻觉。
- 文件文本作为数据处理，不作为系统指令；原页工具只接受已登记文件 ID 和页码，不接受任意文件路径。
- 这是本地单人演示项目，尚未实现用户认证、线上多用户限流、OCR、多政策版本路由、计算报销总额和自动审批。

## 故障排查

| 现象 | 处理 |
|---|---|
| 提示没有密钥 | 本地 `.env` 填写 `DEEPSEEK_API_KEY`；新开终端或重新运行后用 `doctor` 检查布尔状态 |
| AuthenticationError | 核对密钥和 API 服务商；不在聊天中粘贴密钥 |
| APIConnectionError / APITimeoutError | 检查能否连接配置的 DeepSeek API 地址，稍后重试 |
| 模型不存在 / BadRequestError | 官方地址使用 `deepseek-v4-flash`；第三方服务需核对其实际模型 ID 和工具调用兼容性 |
| 模型缓存不存在 | 首次联网时把 `EMBEDDING_LOCAL_ONLY` 设为 `false` 并运行 index |
| 当前配置尚未建立索引 | 更换 Embedding 或切分配置后运行 index；不要混用旧向量 |
| 找到条款但缺少例外 | 查看完整父条款与原页，补充开发题；先定位问题再改提示词 |
| 引用了未检索的编号 | 本次输出已被拦截，可重试或检查结构化输出；不要移除校验来让它“通过” |

## 参考依据

- [LangChain RAG from scratch](https://github.com/langchain-ai/rag-from-scratch)：索引、检索和生成基础。
- [LangChain retrieval-agent-template](https://github.com/langchain-ai/retrieval-agent-template)：索引与问答职责分离。
- [LangChain create_agent](https://docs.langchain.com/oss/python/langchain/agents)：工具循环。
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)：ToolStrategy 与结构化结果。
- [DeepSeek 模型列表](https://api-docs.deepseek.com/quick_start/pricing/)：V4 Flash 模型 ID 与能力。
- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)：显式禁用思考模式的参数。
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)：本地 Embedding 模型。

参考材料用于理解组件和接口；代码未克隆外部 RAG 应用框架。
