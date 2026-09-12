import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    model: str = "deepseek-v4-flash"
    api_base: str = "https://api.deepseek.com"
    embedding_model: str = "BAAI/bge-m3"
    local_only: bool = True
    device: str = "cpu"
    retrieval_mode: str = "hybrid"

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            local_only=os.getenv("EMBEDDING_LOCAL_ONLY", "true").lower() == "true",
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        )

    @property
    def processed(self):
        return self.root / "data/processed"

    @property
    def index(self):
        return self.root / "data/index"
