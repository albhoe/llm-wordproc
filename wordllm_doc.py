from __future__ import annotations
import re
import json
import uuid
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
    hash:str = "" #A hash of the text in this block. Should be updated every time the doc is saved.
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
    blocks: list[Block] = field(default_factory=dict())
    version: int = 1                    # bump when the schema changes
    index:dict[str,int] = {b.id :i for b,i in enumerate(blocks)}

    # ---- helpers -------------------------------------------------------
    def new_block_id(self) -> str:
        return uuid.uuid4().hex[:8]     # short, stable, unique enough for one doc

    def add_paragraph(self, text: str, kind: BlockKind = "paragraph",
                      level: int = 0) -> Block:
        """Adds a new paragraph in the form of a block at the end of the block list"""
        block = Block(id=self.new_block_id(), kind=kind, level=level, runs=[Run(text=text)], hash="")
        self.blocks.append(block)
        return block
    
    def find(self, block_id: str) -> Block | None:
        i = self.index.get(block_id,None)
        return None if i is None else self.blocks[i]

    def save(self,path:Path):
        path = Path(path).with_suffix("").with_suffix(".json")
        tmp = path.with_suffix("").with_suffix(path.suffix + ".tmp")
        # Write to a temp file first, then rename, so a crash can't corrupt the file.
        tmp.write_text(json.dumps(document_to_dict(self), indent=2, ensure_ascii=False),
                   encoding="utf-8")
        tmp.replace(path)

    def get_text(self):
        lines = []
        for b in self.blocks:
            body = runs_to_markup(b.runs)
            if b.kind == "heading":
                body = "#" * max(b.level, 1) + " " + body
            lines.append(body)
        return "\n".join(lines)

    def update_from_text(self,text:str):
        self.blocks = text_to_blocks(text,self)
        self.index = {b.id :i for b,i in enumerate(self.blocks)}


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
            for b in doc.blocks
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
    blocks = []
    for b in data.get("blocks", []):
        blocks.append(Block(
            id=b["id"],
            kind=b["kind"],
            level=b.get("level", 0),
            runs=[Run(**r) for r in b.get("runs", [])],
            hash=b.get("hash",""))
        )
    return WordLLMDocument(title=data.get("title", "Untitled"), manifest=manifest, blocks=blocks,
                    version=version, embedding_model=embedding_model)



def load_document(path: str | Path) -> WordLLMDocument:
    data = json.loads(Path(path).with_suffix("").with_suffix(".json").read_text(encoding="utf-8"))
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
    old_blocks = old_doc.blocks
    old_ids = [b.id for b in old_blocks]
    blocks = []
    for i, line in enumerate(text.split("\n")):
        kind, level = "paragraph", 0
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            kind, level = "heading", hashes
            line = line[hashes:].lstrip()
        block_id = old_ids[i] if i < len(old_ids) else old_doc.new_block_id()
        new_block=(Block(id=block_id, kind=kind, level=level,runs=markup_to_runs(line)))
        if i < len(old_ids):
            new_block.hash = old_blocks[i].hash
        else:
            new_block.hash=""
        blocks.append(new_block)
    return blocks
