"""运行：python -m streamlit run app.py --server.address 127.0.0.1"""
#python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
import os

import streamlit as st

from travel_rag.agent import PolicyService
from travel_rag.config import Settings
from travel_rag.retrieval import PolicyRetriever

st.set_page_config(page_title="上大科研差旅助手", page_icon="📖", layout="centered")#配置整个页面的全局属性
settings = Settings.load()
st.title("上大科研差旅助手")
st.caption("查规则 · 看原文 · 补条件")
st.write("依据《上海大学科研国内差旅费管理办法》（上大内〔2026〕128号），查询科研国内差旅报销规则。") #“通用输出函数”，可以输出文字、DataFrame、图表等


@st.cache_resource    #用于创建并缓存检索器实例
def get_retriever(mode, embedding_model, local_only, device, corpus_stamp):
    """获取检索器"""
    return PolicyRetriever(Settings(embedding_model=embedding_model, local_only=local_only, device=device), mode)


with st.sidebar:  # Streamlit 应用的侧边栏部分
    st.subheader("查询设置")
    task = st.radio("查询方式", ["政策问答", "仅搜索原文"])  #单选按钮组，两个选项
    mode = st.selectbox("检索方式", ["hybrid", "bm25", "vector"],
                        format_func=lambda m: {"hybrid": "混合检索", "bm25": "关键词检索", "vector": "语义检索"}[m])  #下拉选择框
    baseline = st.checkbox("使用普通 RAG 对照模式", value=False, disabled=task != "政策问答")  #复选框
    st.caption("范围：上海大学纵向、横向科研经费的国内差旅。证据来自当前收录文件。")
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        st.info("在项目 .env 中填写 DeepSeek 密钥并重启服务后可问答。关键词搜索原文无需密钥。")
    if st.button("清空对话"):
        st.session_state.messages = []  #清空存储对话历史的列表
        st.rerun() #刷新页面
    st.caption("适合试问：\n\n• 二级教授坐高铁可以选什么座位？\n\n• 横向项目自驾能报汽油费吗？\n\n• 会议包食宿还能领补助吗？")

manifest = settings.processed / "manifest.json"
if not manifest.exists():
    st.info("请先按 README 导入学校 PDF。")
    st.stop()  #立即停止脚本执行
retriever = get_retriever(mode, settings.embedding_model, settings.local_only, settings.device, manifest.stat().st_mtime_ns)


def show_sources(sources): 
    """展示检索到的政策原文出处"""
    for source in sources:
        pages = "、".join(map(str, source["pages"]))  #通过str函数映射，3->’3‘
        with st.expander(f"{source['article']} · PDF 第 {pages} 页"):
            st.caption(f"{source['source']} · 证据编号 {source['id']}")
            st.text(source["text"])
            if "raw_text" in source:
                st.caption("页面原始提取文本（表格可能错位，金额请对照 PDF）")
                st.text(source["raw_text"])


st.session_state.setdefault("messages", [])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("例如：会议统一安排住宿但费用自理，补助怎么算？", max_chars=2000):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages
               if not m.get("search_only")]
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question, "search_only": task == "仅搜索原文"})
    with st.chat_message("assistant"):
        try:
            if task == "仅搜索原文":
                with st.spinner("正在搜索…"):
                    sources = retriever.search(question)
                text = "找到以下候选条款，请展开核对适用条件。" if sources else "没有找到匹配条款，可以换一种表达。"
                st.markdown(text)
                show_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": text,
                                                  "sources": sources, "search_only": True})
            else:
                service = PolicyService(retriever, settings)
                with st.status("正在核对政策依据…", expanded=True) as status:
                    for event in service.stream(question, history, baseline):
                        if event["type"] == "progress":
                            st.write(event["message"])
                        elif event["type"] == "tool":
                            st.write("已补查相关条款" if event["tool"] == "search_policy" else "已核对原文页面")
                        elif event["type"] == "answer":
                            result = event["result"]
                    status.update(label="已完成核对", state="complete", expanded=False)
                st.markdown(result["text"])
                show_sources(result["sources"])
                with st.expander("本次查询记录"):
                    st.write(f"用时 {result['seconds']} 秒；补查工具 {len(result['trace'])} 次。")
                    if result["usage"]:
                        st.json(result["usage"])
                    st.caption("引用编号已经校验；这不等于自动证明每一项结论都正确，请结合原文核对。")
                st.session_state.messages.append({"role": "assistant", "content": result["text"], "sources": result["sources"]})
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))
            st.session_state.messages.pop()
        except Exception as error:
            st.error(f"本次查询未完成（{type(error).__name__}）。请检查密钥、网络或本地索引；可按 README 排查。")
            st.session_state.messages.pop()
