from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("PENSION_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT

    @property
    def raw_markdown(self) -> Path:
        return self.project_root / "raw_markdown"

    @property
    def ontology(self) -> Path:
        return self.project_root / "ontology"

    @property
    def processed(self) -> Path:
        return self.project_root / "data" / "processed"

    def assert_corpus_path(self, path: Path) -> Path:
        """Reject inputs outside raw_markdown, even if a caller passes another path."""
        resolved = path.resolve()
        allowed = self.raw_markdown.resolve()
        if resolved != allowed:
            raise ValueError(f"입력은 raw_markdown만 허용됩니다: {resolved}")
        return resolved
