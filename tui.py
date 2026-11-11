import re
import sys

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, TextArea, Static

from llm_proc_doc import Block, LLMProcDocument, Run, load_document, save_document, text_to_blocks, document_to_text
from llm_functions import WordProcessorAssistant




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
        yield Static("AI response will appear here.", id="answer")
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

    @on(Input.Submitted, "#question")
    def ask_question(self, event: Input.Submitted) -> None:
        self.query_one("#answer").update("Thinking...")
        question = event.value.strip()
        if not question:
            return
        answer = self.ai.answer_question(question, context="No context in this document.")
        self.query_one("#answer").update(answer)

if __name__ == "__main__":
    NotepadApp(sys.argv[1] if len(sys.argv) > 1 else "draft.json").run()