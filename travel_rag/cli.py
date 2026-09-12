import argparse
import json
import os
import sys
from pathlib import Path

from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="上海大学差旅政策 RAG 教学项目")
    commands = parser.add_subparsers(dest="command", required=True)
    ingestion = commands.add_parser("ingest", help="解析并导入已核对的 PDF")
    ingestion.add_argument("pdf", type=Path)
    commands.add_parser("index", help="建立或复用本地向量索引")
    commands.add_parser("doctor", help="检查配置，不显示密钥")
    for name in ("search", "ask", "chat"):
        command = commands.add_parser(name)
        if name != "chat":
            command.add_argument("question")
        command.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default=None)
        if name != "search":
            command.add_argument("--baseline", action="store_true", help="使用单次生成的普通 RAG")
    evaluation = commands.add_parser("eval", help="评测检索；加 --generate 可调用模型导出人工评阅样本")
    evaluation.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    evaluation.add_argument("--split", choices=["dev", "test"], default="dev")
    evaluation.add_argument("--generate", action="store_true")
    evaluation.add_argument("--baseline", action="store_true")
    evaluation.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    settings = Settings.load()
    try:
        if args.command == "doctor":
            print(json.dumps({"model": settings.model, "api_base": settings.api_base,
                              "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
                              "embedding_model": settings.embedding_model,
                              "local_only": settings.local_only,
                              "pdf_ingested": (settings.processed / "manifest.json").exists(),
                              "index_markers": len(list(settings.index.glob("*.json")))},
                             ensure_ascii=False, indent=2))
            return
        if args.command == "ingest":
            from .ingest import ingest
            result = ingest(args.pdf, settings.root)
        elif args.command == "eval":
            from .evaluate import evaluate_retrieval, export_answers
            if args.generate:
                if not 1 <= args.limit <= 40:
                    raise ValueError("limit 必须是 1～40。")
                if not os.getenv("DEEPSEEK_API_KEY", "").strip():
                    raise ValueError("请先在 .env 配置 DEEPSEEK_API_KEY。")
                result = {"output": str(export_answers(settings, args.mode, args.split, args.limit, args.baseline))}
            else:
                summary, path = evaluate_retrieval(settings, args.mode, args.split)
                result = {**summary, "output": str(path)}
        else:
            from .retrieval import PolicyRetriever
            retriever = PolicyRetriever(settings, getattr(args, "mode", None))
            if args.command == "index":
                result = retriever.build_index()
            elif args.command == "search":
                result = retriever.search(args.question)
            else:
                from .agent import PolicyService
                service = PolicyService(retriever, settings)
                history = []
                while True:
                    question = input("你（输入 /quit 退出）：").strip() if args.command == "chat" else args.question
                    if question == "/quit":
                        return
                    for event in service.stream(question, history, args.baseline):
                        if event["type"] == "progress":
                            print(event["message"])
                        elif event["type"] == "tool":
                            print(f"已完成工具：{event['tool']}")
                        elif event["type"] == "answer":
                            result = event["result"]
                    print(result["text"])
                    for source in result["sources"]:
                        print(f"  [{source['id']}] {source['article']}，PDF第{source['pages']}页")
                    if args.command == "ask":
                        return
                    history.extend([{"role": "user", "content": question},
                                    {"role": "assistant", "content": result["text"]}])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as error:
        print(f"需要处理：{error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已退出。")
    except Exception as error:
        # SDK 原始报错可能包含请求正文，不直接打印。
        print(f"运行未完成（{type(error).__name__}）。请检查网络、模型配置或索引；详见 README 故障排查。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
