import re
import sys

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, TextArea, Static, Markdown
from textual.containers import VerticalScroll
import argparse
from pathlib import Path

from llm_proc_doc import Block, LLMProcDocument, Run, load_document, save_document, text_to_blocks, document_to_text
from llm_functions import WordProcessorAssistant

parser = argparse.ArgumentParser(description="Open a document in the LLM Word Processor")
parser.add_argument("path", type=str, help="Path to the document file to be opened.")
parser.add_argument("-n", "--new", action="store_true", help="Create a new document instead of opening an existing one.")


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
        self.ai = WordProcessorAssistant(document_path=path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Manifest: {self.doc.manifest}")
        yield TextArea(document_to_text(self.doc), id="editor")
        yield Input(placeholder="Questions for AI?", id="question")
        with VerticalScroll(id="answer-scroll"):
            yield Markdown("AI response will appear here.", id="answer")
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
        self.ai.document = self.doc  # Update the AI assistant's document reference
        self.notify("Saved")

    @on(Input.Submitted, "#question")
    def ask_question(self, event: Input.Submitted) -> None:
        self.query_one("#answer").update("Thinking...")
        question = event.value.strip()
        if not question:
            return
        if not (highlight := self.query_one(TextArea).selected_text):
            highlight = "The user has not highlighted any specific text."
        answer = self.ai.answer_question(question, highlight=highlight)
        self.query_one("#answer").update(answer)



if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path 
    if new_flag := args.new:
        if Path(path).exists():
            raise FileExistsError(f"File {path} already exists. Use a different path or remove the file.")
        else:
            doc = LLMProcDocument(title="New Document", manifest="A general purpose document", blocks=[])
            save_document(doc, path)
    else:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist. Use the --new flag to create a new document.")
    NotepadApp(path).run()