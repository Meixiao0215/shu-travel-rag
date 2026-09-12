"""用 HTTP 替身核对真实 ChatDeepSeek 适配器发出的请求，不调用付费 API。"""
import json

import httpx
import pytest
from langchain_deepseek import ChatDeepSeek

from travel_rag.agent import PolicyService
from travel_rag.config import ROOT, Settings
from travel_rag.retrieval import PolicyRetriever


@pytest.mark.parametrize("baseline", [False, True])
def test_deepseek_adapter_request_and_response(tmp_path, baseline):
    if not (ROOT / "data/processed/manifest.json").exists():
        pytest.skip("先导入提供的 PDF。")
    retriever = PolicyRetriever(Settings(), "bm25")
    evidence_id = next(a["id"] for a in retriever.articles.values() if a["article"] == "第二十四条")
    requests = []
    answer = {"status": "answered", "claims": [{"text": "自驾报销需区分纵向和横向科研项目。",
              "evidence_ids": [evidence_id]}], "message": "", "questions": []}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path.endswith("/chat/completions")
        assert body["model"] == "deepseek-v4-flash"
        assert body["thinking"] == {"type": "disabled"}
        if baseline:
            assert body["response_format"] == {"type": "json_object"}
            message = {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
            finish = "stop"
        else:
            functions = [t["function"]["name"] for t in body["tools"]]
            assert "search_policy" in functions and "PolicyAnswer" in functions
            if len(requests) == 1:
                name, args = "search_policy", {"query": "横向自驾汽油费"}
            else:
                assert any(m["role"] == "tool" for m in body["messages"])
                name, args = "PolicyAnswer", answer
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{len(requests)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
            finish = "tool_calls"
        return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion",
            "created": 1, "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatDeepSeek(model="deepseek-v4-flash", api_key="fake-local-test-key",
                             http_client=client, max_retries=0,
                             extra_body={"thinking": {"type": "disabled"}})
        service = PolicyService(retriever, Settings(root=tmp_path), model=model)
        result = service.ask("横向项目自驾汽油费能否报销？", baseline=baseline)
    assert result["sources"][0]["id"] == evidence_id
    assert len(requests) == (1 if baseline else 2)
