import re
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TextArea, Static

from document import Block, Document, Run, load_document, save_document


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


def document_to_text(doc: Document) -> str:
    """One paragraph per line; headings get a leading '#'."""
    lines = []
    for b in doc.blocks:
        body = runs_to_markup(b.runs)
        if b.kind == "heading":
            body = "#" * max(b.level, 1) + " " + body
        lines.append(body)
    return "\n".join(lines)


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


# ---- the app -------------------------------------------------------------

class NotepadApp(App):
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+b", "toggle_bold", "Bold"),
        Binding("ctrl+i", "toggle_italic", "Italic"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.doc = load_document(path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Manifest: {self.doc.manifest}")
        yield TextArea(document_to_text(self.doc), id="editor")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.doc.title
        self.query_one(TextArea).focus()

    def _wrap_selection(self, marker: str) -> None:
        editor = self.query_one(TextArea)
        selected = editor.selected_text
        if not selected:
            return
        editor.replace(f"{marker}{selected}{marker}", *editor.selection)

    def action_toggle_bold(self) -> None:
        self._wrap_selection("**")

    def action_toggle_italic(self) -> None:
        self._wrap_selection("*")

    def action_save(self) -> None:
        text = self.query_one(TextArea).text
        self.doc.blocks = text_to_blocks(text, self.doc)
        save_document(self.doc, self.path)
        self.notify("Saved")


if __name__ == "__main__":
    NotepadApp(sys.argv[1] if len(sys.argv) > 1 else "draft.json").run()