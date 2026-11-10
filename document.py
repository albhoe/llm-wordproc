from __future__ import annotations
import re

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

BlockKind = Literal["paragraph", "heading", "list_item"]


@dataclass
class Run:
    #Contains a sequence of characters with the same formatting (bold/italic).
    text: str
    bold: bool = False
    italic: bool = False


@dataclass
class Block:
    #Contains a list of runs.
    id: str
    kind: BlockKind
    level: int = 0                      # heading level or list depth
    runs: list[Run] = field(default_factory=list)
    @property
    def text(self) -> str:
        """Plain text of the block, ignoring formatting."""
        return "".join(r.text for r in self.runs)


@dataclass
class Document:
    title: str = "Untitled"
    manifest: str = "A document for general use." #Explains the overall "vibe" of the document for the LLM.
    blocks: list[Block] = field(default_factory=list)
    version: int = 1                    # bump when the schema changes

    # ---- helpers -------------------------------------------------------
    def new_block_id(self) -> str:
        return uuid.uuid4().hex[:8]     # short, stable, unique enough for one doc

    def add_paragraph(self, text: str, kind: BlockKind = "paragraph",
                      level: int = 0) -> Block:
        block = Block(id=self.new_block_id(), kind=kind, level=level,
                      runs=[Run(text=text)])
        self.blocks.append(block)
        return block

    def find(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)


# ---- JSON serialization -------------------------------------------------

SCHEMA_VERSION = 1


def document_to_dict(doc: Document) -> dict:
    return {
        "schema": "llm-wordproc",
        "manifest": doc.manifest,
        "version": SCHEMA_VERSION,
        "title": doc.title,
        "blocks": [
            {
                "id": b.id,
                "kind": b.kind,
                "level": b.level,
                "runs": [asdict(r) for r in b.runs],
            }
            for b in doc.blocks
        ],
    }


def document_from_dict(data: dict) -> Document:
    if data.get("schema") != "llm-wordproc":
        raise ValueError("Not an llm-wordproc document")
    manifest = data.get("manifest", "")
    version = data.get("version", 1)
    if version > SCHEMA_VERSION:
        raise ValueError(f"Document version {version} is newer than supported")

    blocks = [
        Block(
            id=b["id"],
            kind=b["kind"],
            level=b.get("level", 0),
            runs=[Run(**r) for r in b.get("runs", [])],
        )
        for b in data.get("blocks", [])
    ]
    return Document(title=data.get("title", "Untitled"), manifest=manifest, blocks=blocks,
                    version=version)


def save_document(doc: Document, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write to a temp file first, then rename, so a crash can't corrupt the file.
    tmp.write_text(json.dumps(document_to_dict(doc), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


def load_document(path: str | Path) -> Document:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return document_from_dict(data)

# ---- conversion between model and editor text --------------------------

def runs_to_markup(runs: list[Run]) -> str:
    out = []
    for run in runs:
        text = run.text
        if run.bold and run.italic:
            out.append(f"***{text}***")
        elif run.bold:
            out.append(f"**{text}**")
        elif run.italic:
            out.append(f"*{text}*")
        else:
            out.append(text)
    return "".join(out)


TOKEN = re.compile(r"(\*\*\*.+?\*\*\*|\*\*.+?\*\*|\*.+?\*)")


def markup_to_runs(text: str) -> list[Run]:
    runs = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("***") and part.endswith("***") and len(part) > 6:
            runs.append(Run(text=part[3:-3], bold=True, italic=True))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            runs.append(Run(text=part[2:-2], bold=True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs.append(Run(text=part[1:-1], italic=True))
        else:
            runs.append(Run(text=part))
    return runs


def text_to_blocks(text: str, old_doc: Document) -> list[Block]:
    """Rebuild blocks from text. Reuses old IDs by line position when possible."""
    old_ids = [b.id for b in old_doc.blocks]
    blocks = []
    for i, line in enumerate(text.split("\n")):
        kind, level = "paragraph", 0
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            kind, level = "heading", hashes
            line = line[hashes:].lstrip()
        block_id = old_ids[i] if i < len(old_ids) else old_doc.new_block_id()
        blocks.append(Block(id=block_id, kind=kind, level=level,
                            runs=markup_to_runs(line)))
    return blocks

def document_to_text(doc: Document) -> str:
    """One paragraph per line; headings get a leading '#'."""
    lines = []
    for b in doc.blocks:
        body = runs_to_markup(b.runs)
        if b.kind == "heading":
            body = "#" * max(b.level, 1) + " " + body
        lines.append(body)
    return "\n".join(lines)