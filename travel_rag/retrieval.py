"""学习顺序 2：中文 BM25 + 向量检索，用 RRF 融合后返回完整条款。"""
import hashlib
import json
import logging
import re
from functools import lru_cache
from threading import Lock

import jieba
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .config import Settings
from .ingest import read_json, write_json

jieba.setLogLevel(logging.ERROR)
STOPWORDS = set("的 了 吗 呢 啊 我 你 请 请问 可以 是否 怎么 如何 什么 多少 能 在 是 和 与 及 有 要".split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in jieba.lcut_for_search(text)
            if re.search(r"[\w\u4e00-\u9fff]", t) and t not in STOPWORDS]


class LocalEmbeddings(Embeddings):
    """实现 LangChain Embeddings 接口，明确区分文档与查询编码。"""
    def __init__(self, model: str, device: str, local_only: bool):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.encoder = SentenceTransformer(model, device=device, local_files_only=local_only)
        self.lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self.lock:
            return self.encoder.encode(texts, batch_size=8, normalize_embeddings=True,
                                       show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE-M3 不需要指令前缀；若更换为 BGE 中文 v1.5，则使用官方建议前缀。
        if "bge-" in self.model_name.lower() and "zh-v1.5" in self.model_name.lower():
            text = "为这个句子生成表示以用于检索相关文章：" + text
        return self.embed_documents([text])[0]


@lru_cache(maxsize=2)
def embeddings(model: str, device: str, local_only: bool):
    return LocalEmbeddings(model, device, local_only)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class PolicyRetriever:
    def __init__(self, settings: Settings, mode: str | None = None):
        self.settings = settings
        self.mode = mode or settings.retrieval_mode
        if self.mode not in ("bm25", "vector", "hybrid"):
            raise ValueError("检索模式必须是 bm25 / vector / hybrid")
        if not (settings.processed / "manifest.json").exists():
            raise ValueError("请先运行 python -m travel_rag.cli ingest <PDF路径>")
        self.manifest = read_json(settings.processed / "manifest.json")
        self.articles = {a["id"]: a for a in read_json(settings.processed / "articles.json")}
        self.pages = {p["page_number"]: p for p in read_json(settings.processed / "pages.json")}
        self.chunks = read_json(settings.processed / "chunks.json")
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        signature = json.dumps({"model": settings.embedding_model, "chunks": self.chunks,
                                "embedding_version": 1}, sort_keys=True, ensure_ascii=False)
        self.fingerprint = hashlib.sha256(signature.encode()).hexdigest()
        self.collection_name = f"policy-{self.fingerprint[:20]}"
        self.vector_store = None

    def _vectors(self, building=False):
        if self.vector_store is not None:
            return self.vector_store
        marker = self.settings.index / f"{self.fingerprint}.json"
        if not building and not marker.exists():
            raise ValueError("当前文档或 Embedding 配置尚未建立索引，请运行 python -m travel_rag.cli index")
        from chromadb.config import Settings as ChromaSettings
        from langchain_chroma import Chroma
        embedding = embeddings(self.settings.embedding_model, self.settings.device, self.settings.local_only)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.settings.index / "chroma"),
            embedding_function=embedding,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
        return self.vector_store

    def build_index(self):
        store = self._vectors(building=True)
        existing = set(store.get()["ids"])
        missing = [c for c in self.chunks if c["id"] not in existing]
        for offset in range(0, len(missing), 16):
            batch = missing[offset:offset + 16]
            store.add_documents(
                [Document(page_content=c["text"], metadata={"parent_id": c["parent_id"], "chunk_id": c["id"]})
                 for c in batch], ids=[c["id"] for c in batch],
            )
        if set(store.get()["ids"]) != {c["id"] for c in self.chunks}:
            raise RuntimeError("索引条目不完整，尚未标记为可用。")
        result = {"fingerprint": self.fingerprint, "collection": self.collection_name,
                  "embedding_model": self.settings.embedding_model,
                  "count": len(self.chunks), "added": len(missing)}
        write_json(self.settings.index / f"{self.fingerprint}.json", result)
        return result

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("检索问题需要 1～2000 个字符。")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 需要在 1～10 之间。")
        rankings = []
        candidates = min(max(top_k * 4, 20), len(self.chunks))
        if self.mode in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            rankings.append([self.chunks[i]["id"] for i in order[:candidates] if scores[i] > 0])
        if self.mode in ("vector", "hybrid"):
            docs = self._vectors().similarity_search(query, k=candidates)
            rankings.append([d.metadata["chunk_id"] for d in docs])
        # 先按子块融合，再按父条款去重，返回完整父条款以保留例外条件。
        results, seen = [], set()
        for chunk_id, score in reciprocal_rank_fusion(rankings):
            parent_id = self.chunk_by_id[chunk_id]["parent_id"]
            if parent_id not in seen:
                results.append({**self.articles[parent_id], "retrieval_score": score})
                seen.add(parent_id)
            if len(results) == top_k:
                break
        return results

    def read_page(self, doc_id: str, page_number: int):
        if doc_id != self.manifest["doc_id"]:
            raise ValueError("未知文件编号。只能读取知识库中的文件。")
        if page_number not in self.pages:
            raise ValueError("页码超出范围。")
        page = self.pages[page_number]
        return {"id": f"{doc_id}:p{page_number:03d}", "doc_id": doc_id,
                "source": self.manifest["source"], "article": "原文页面（含表格整理）",
                "pages": [page_number], "text": page["text"], "raw_text": page["raw_text"]}

    def scope_evidence(self):
        return [a for a in self.articles.values()
                if a["article"] in ("第二条", "第三条", "第三十三条", "发文通知")]
