from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, TextArea, Static, Markdown
from textual.containers import VerticalScroll
import argparse
from pathlib import Path

from wordllm_doc import WordLLMDocument
from llm_functions import WordProcessorAssistant

parser = argparse.ArgumentParser(description="Open a document in the LLM Word Processor")
parser.add_argument("path", type=str, help="Path to the document file to be opened.")
parser.add_argument("-n", "--new", action="store_true", help="Create a new document instead of opening an existing one.")
parser.add_argument("-v","--verbose",type=int,default=0,help="Enable verbose output for debugging purposes. Outputs the internal messages sliced to the specified limit. Numbers less than 1 result in no verbose output.")


# ---- the app -------------------------------------------------------------

class NotepadApp(App):
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+b", "toggle_bold", "Bold"),
        Binding("ctrl+i", "toggle_italic", "Italic"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, path: str, verbose: bool = False) -> None:
        super().__init__()
        self.path = path
        self.verbose = verbose
        self.ai = WordProcessorAssistant(document_path=path+".json", verbose=verbose)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Manifest: {self.ai.doc.manifest}", id="manifest")
        yield TextArea(self.ai.doc.get_text(), id="editor")
        yield Input(placeholder="Questions for AI?", id="question")
        with VerticalScroll(id="answer-scroll"):
            yield Markdown("AI response will appear here.", id="answer")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.ai.doc.title
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
        self.ai.save_doc_and_retriever(text)
        self.notify("Saved")

    def update_all(self, ai_response: str | None = None) -> None:
        self.title = self.ai.doc.title
        self.query_one("#manifest").update(f"Manifest: {self.ai.doc.manifest}")
        self.query_one('#editor').text = self.ai.doc.get_text()
        if ai_response:
            self.query_one('#answer').update(ai_response)
        else:
            self.query_one('#answer').update("AI response will appear here.")

    @on(Input.Submitted, "#question")
    def ask_question(self, event: Input.Submitted) -> None:
        self.query_one("#answer").update("Thinking...")
        question = event.value.strip()
        if not question:
            return
        highlight = self.query_one(TextArea).selected_text
        try:
            answer = self.ai.answer_question(question, highlight=highlight)
        except Exception as e:
            answer = f"Sorry, there was an exception with your request: {e}"
        self.update_all(ai_response=answer)




if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    memory_path = Path(str(path)).with_suffix(".json")
    index_path = Path(str(path)+"_index").with_suffix(".npy")
    if new_flag := args.new:
        if Path(memory_path).exists() or Path(index_path).exists():
            raise FileExistsError(f"File {path} already exists. Use a different path or manually remove the file.")
        else:
            doc = WordLLMDocument(title="New Document", manifest="A general purpose document", blocks=dict())
            doc.save(path)
    else:
        if not (memory_path.exists()):
            raise FileNotFoundError(f"{path} is missing one or both files. Use the --new flag to create a new document.")
    NotepadApp(path, verbose=args.verbose).run()