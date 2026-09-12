"""学习顺序 3：可追溯的回答协议，每个结论都绑定证据。"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    text: str = Field(min_length=1, description="一项基于证据的结论，写明适用条件与例外。")
    evidence_ids: list[str] = Field(min_length=1, description="本轮工具或初始证据中确实存在的编号。")


class PolicyAnswer(BaseModel):
    status: Literal["answered", "need_clarification", "insufficient_evidence"]
    claims: list[Claim] = Field(default_factory=list, description="有证据支持的结论；不能凭常识补充政策。")
    message: str = Field(default="", description="仅说明缺失条件或证据不足，不能在此添加无引用的政策结论。")
    questions: list[str] = Field(default_factory=list, description="只追问对本题适用条款有影响的条件。")

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "answered" and not self.claims:
            raise ValueError("answered 必须有带证据的结论。")
        if self.status == "need_clarification" and not self.questions:
            raise ValueError("need_clarification 必须给出追问。")
        if self.status == "insufficient_evidence" and not self.message:
            raise ValueError("证据不足时必须说明缺少什么。")
        return self


def validate_evidence(answer: PolicyAnswer, evidence: dict[str, dict]):
    requested = {i for claim in answer.claims for i in claim.evidence_ids}
    unknown = requested - evidence.keys()
    if unknown:
        raise ValueError("回答引用了本轮未检索到的证据编号。")
    return [evidence[i] for i in sorted(requested)]


def render_answer(answer: PolicyAnswer) -> str:
    parts = []
    if answer.status != "answered" and answer.message:
        parts.append(answer.message)
    for claim in answer.claims:
        references = " ".join(f"[{i}]" for i in claim.evidence_ids)
        parts.append(f"{claim.text} {references}")
    if answer.questions:
        parts.append("请补充：\n" + "\n".join(f"- {q}" for q in answer.questions))
    return "\n\n".join(parts)
