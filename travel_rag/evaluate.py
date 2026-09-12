"""检索评测与生成结果导出。指标不把模型自评当作真实正确率。"""
import time

from .ingest import read_json, write_json
from .retrieval import PolicyRetriever


def evaluate_retrieval(settings, mode="hybrid", split="dev", top_k=5):
    cases = read_json(settings.root / "eval/cases.json")
    retriever = PolicyRetriever(settings, mode)
    results = []
    # 追问、无答案、多轮问题需要生成阶段评测，不混入单轮检索指标。
    for case in cases:
        if case["split"] != split or case["expected_status"] != "answered" or case.get("history"):
            continue
        started = time.monotonic()
        found = retriever.search(case["question"], top_k=top_k)
        labels = [r["article"] for r in found]
        expected = set(case["expected_articles"])
        hits = expected.intersection(labels)
        rank = next((i for i, label in enumerate(labels, 1) if label in expected), None)
        results.append({"id": case["id"], "question": case["question"],
                        "expected": sorted(expected), "retrieved": labels,
                        "recall": len(hits) / len(expected),
                        "all_evidence_found": hits == expected,
                        "reciprocal_rank": 1 / rank if rank else 0,
                        "seconds": round(time.monotonic() - started, 3)})
    if not results:
        raise ValueError("此划分没有可评测的单轮问题。")
    report = {"mode": mode, "split": split, "top_k": top_k, "count": len(results),
              "mean_evidence_recall": sum(r["recall"] for r in results) / len(results),
              "all_evidence_rate": sum(r["all_evidence_found"] for r in results) / len(results),
              "mrr": sum(r["reciprocal_rank"] for r in results) / len(results),
              "embedding_model": settings.embedding_model if mode != "bm25" else None,
              "corpus_sha256": retriever.manifest["sha256"], "results": results,
              "note": "人工从单份PDF设计的小样本评测；仅测检索，不是最终答案正确率。"}
    path = settings.root / f"eval/reports/{split}_{mode}.json"
    write_json(path, report)
    return {k: v for k, v in report.items() if k != "results"}, path


def export_answers(settings, mode="hybrid", split="dev", limit=5, baseline=False):
    from .agent import PolicyService
    service = PolicyService(PolicyRetriever(settings, mode), settings)
    cases = [c for c in read_json(settings.root / "eval/cases.json") if c["split"] == split][:limit]
    outputs = []
    for case in cases:
        try:
            result = service.ask(case["question"], case.get("history"), baseline=baseline)
            outputs.append({"case": case, "result": result,
                            "review": {"status_correct": None, "claims_supported": None,
                                       "conditions_complete": None, "notes": "待人工对照原文评阅"}})
        except Exception as error:
            outputs.append({"case": case, "error_type": type(error).__name__})
    name = "baseline" if baseline else "agent"
    path = settings.root / f"eval/reports/{split}_{mode}_{name}_answers.json"
    write_json(path, outputs)
    return path
