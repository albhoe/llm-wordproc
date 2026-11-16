from __future__ import annotations
import re
import json
import uuid
import numpy
from sentence_transformers import SentenceTransformer
import hashlib
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
    level: int = 0  
    hash:str = "" #A hash of the text in this block. Updates every time the doc is saved.
    runs: list[Run] = field(default_factory=list)
    @property
    def text(self) -> str:
        """Plain text of the block, ignoring formatting."""
        return "".join(r.text for r in self.runs)
    def compute_hash(self) -> str:
        payload = repr(self.text)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    def refresh_hash(self) -> tuple[str, bool]:
        """Recompute the hash. Returns the hash, and a boolean denoting if the block changed."""
        new = self.compute_hash()
        changed = (new != self.hash)
        self.hash = new
        return new, changed

@dataclass
class WordLLMDocument:
    title: str = "Untitled"
    manifest: str = "A document for general use." #Explains the overall "vibe" of the document for the LLM.
    embedding_model:str = "all-MiniLM-L6-v2"
    blocks: dict[str, Block] = field(default_factory=dict())
    version: int = 1                    # bump when the schema changes

    # ---- helpers -------------------------------------------------------
    def new_block_id(self) -> str:
        return uuid.uuid4().hex[:8]     # short, stable, unique enough for one doc

    def add_paragraph(self, text: str, kind: BlockKind = "paragraph",
                      level: int = 0) -> Block:
        block = Block(id=self.new_block_id(), kind=kind, level=level,
                      runs=[Run(text=text)])
        self.blocks[block.id] = block
        return block
    
    def find(self, block_id: str) -> Block | None:
        return self.blocks.get(block_id)

    def save(self,text:str,path:Path):
        save_document(self,path)

    def get_text(self):
        return document_to_text(self)


# ---- JSON serialization -------------------------------------------------

SCHEMA_VERSION = 1


def document_to_dict(doc: WordLLMDocument) -> dict:
    return {
        "schema": "llm-wordproc",
        "manifest": doc.manifest,
        "embedding_model": doc.embedding_model,
        "version": SCHEMA_VERSION,
        "title": doc.title,
        "blocks": [
            {
                "id": b.id,
                "kind": b.kind,
                "level": b.level,
                "runs": [asdict(r) for r in b.runs],
                "hash": b.hash
            }
            for b in doc.blocks.values()
        ],
    }


def document_from_dict(data: dict) -> WordLLMDocument:
    if data.get("schema") != "llm-wordproc":
        raise ValueError("Not an llm-wordproc document")
    manifest = data.get("manifest", "")
    version = data.get("version", 1)
    embedding_model = data.get("embedding_model","all-MiniLM-L6-v2")
    if version > SCHEMA_VERSION:
        raise ValueError(f"Document version {version} is newer than supported")
    blocks = dict()
    for b in data.get("blocks", []):
        blocks[b["id"]] = Block(
            id=b["id"],
            kind=b["kind"],
            level=b.get("level", 0),
            runs=[Run(**r) for r in b.get("runs", [])],
            hash=b.get("hash","")
        )
    return WordLLMDocument(title=data.get("title", "Untitled"), manifest=manifest, blocks=blocks,
                    version=version, embedding_model=embedding_model)


def save_document(doc: WordLLMDocument, path: str | Path) -> None:
    path = Path(path).with_suffix(".json")
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write to a temp file first, then rename, so a crash can't corrupt the file.
    tmp.write_text(json.dumps(document_to_dict(doc), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


def load_document(path: str | Path) -> WordLLMDocument:
    data = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
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


def text_to_blocks(text: str, old_doc: WordLLMDocument) -> dict[str,Block]:
    """Rebuild blocks from text. Reuses old IDs by line position when possible."""
    old_ids = [b.id for b in old_doc.blocks.values()]
    blocks = dict()
    for i, line in enumerate(text.split("\n")):
        kind, level = "paragraph", 0
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            kind, level = "heading", hashes
            line = line[hashes:].lstrip()
        block_id = old_ids[i] if i < len(old_ids) else old_doc.new_block_id()
        new_block=(Block(id=block_id, kind=kind, level=level,runs=markup_to_runs(line)))
        blocks[block_id]=new_block
    return blocks

def document_to_text(doc: WordLLMDocument) -> str:
    """One paragraph per line; headings get a leading '#'."""
    lines = []
    for b in doc.blocks.values():
        body = runs_to_markup(b.runs)
        if b.kind == "heading":
            body = "#" * max(b.level, 1) + " " + body
        lines.append(body)
    return "\n".join(lines)