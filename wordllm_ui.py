from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, TextArea, Static, Markdown
from textual.containers import VerticalScroll
from textual.widgets.text_area import Selection
from typing import Optional
import argparse
from pathlib import Path
import traceback

from llm_functions import WordProcessorAssistant

parser = argparse.ArgumentParser(description="Open a document in the LLM Word Processor")
parser.add_argument("path", type=str, help="Path to the document file to be opened.")
parser.add_argument("-n", "--new", action="store_true", help="Create a new document instead of opening an existing one.")
parser.add_argument("-v","--verbose",type=int,default=0,help="Enable verbose output for debugging purposes. Outputs the internal messages sliced to the specified limit. Numbers less than 1 result in no verbose output.")


# ---- the app -------------------------------------------------------------

class NotepadApp(App):
    CSS = """
    TextArea .text-area--selection {
        background: yellow;
        color: black;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+b", "toggle_bold", "Bold"),
        Binding("ctrl+i", "toggle_italic", "Italic"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+comma", "prev", "Previous"),
        Binding("ctrl+full_stop", "next", "Next"),
    ]

    def __init__(self, path: str, verbose: bool = False, fresh = False) -> None:
        super().__init__()
        self.path = Path(path).with_suffix("").with_suffix(".json")
        print(self.path)
        self.verbose = verbose
        self.ai = WordProcessorAssistant(document_path=self.path, verbose=verbose, fresh = fresh)

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
        synced_blocks = self.ai.save_doc_and_retriever(text)
        self.notify(f"Saved. {synced_blocks} synchronized with retriever.")

    def action_prev(self) -> None:
        if (i:=self.ai.fsiterator_prev()) is not None:
            self.jump_to(torow=i)
            self.notify(f"Result {self.ai.fsiterator.progress()}")

    def action_next(self) -> None:
        if (i:=self.ai.fsiterator_next()) is not None:
            self.jump_to(torow=i)
            self.notify(f"Result {self.ai.fsiterator.progress()}")

    def jump_to(self, torow:int, tocol:Optional[int]=None, fromrow:Optional[int]=None, fromcol:Optional[int]=None,):
            editor = self.query_one('#editor',TextArea)
            if fromcol and fromcol:
                #If there is a starting position, begin with the cursor there
                editor.move_cursor((fromcol,fromcol))

            if tocol:
                #If the ending position is specified, move the cursor there selecting and centering
                editor.move_cursor((torow,0))
                editor.move_cursor((torow,tocol),select=True,center=True)
            else:
                #If no ending position is specified, go to the end of the row 
                editor.select_line(torow)
            editor.focus()
            

    def update_all(self, ai_response: str | None = None) -> None:
        self.title = self.ai.doc.title
        self.query_one("#manifest").update(f"Manifest: {self.ai.doc.manifest}")
        editor = self.query_one('#editor',TextArea) 
        editor.history.checkpoint()
        editor.replace(start=(0,0),end=editor.document.end, insert=self.ai.doc.get_text())
        editor = self.query_one('#editor')
        if ai_response:
            self.query_one('#answer').update(ai_response)
        else:
            self.query_one('#answer').update("AI response will appear here.")

        self.action_next()

    @on(Input.Submitted, "#question")
    def ask_question(self, event: Input.Submitted) -> None:
        self.notify("Thinking...")
        question = event.value.strip()
        if not question:
            return
        editor = self.query_one(TextArea)
        selected = editor.selected_text
        r1,c1 = editor.selection.start
        r2,c2 = editor.selection.end
        #self.notify(f"selected {startrow},{startcol} to {end}, select text: {select}")
        try:
            answer = self.ai.answer_question(question, selected=selected, selected_location=((r1,c1),(r2,c2)))
        except Exception as e:
            if self.verbose > 0:
                answer =  traceback.print_exc()
            else:
                answer = f"Sorry, there was an exception with your request:{type(e).__name__}: {e!r}"
        self.update_all(ai_response=answer)
        #Reset cursor to their initial position.
        editor.move_cursor(location=(r1,c1),center=True)
        return




if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    memory_path = Path(str(path)).with_suffix("").with_suffix(".json")
    index_path = Path(str(path)+"_index").with_suffix("").with_suffix(".npy")
    fresh = False
    if args.new:
        if Path(memory_path).exists() or Path(index_path).exists():
            raise FileExistsError(f"File {path} already exists. Use a different path or manually remove the file.")
        else:
            if args.verbose > 0:
                print("Creating a fresh document")
            fresh = True
    else:
        if not (memory_path.exists()):
            raise FileNotFoundError(f"{path} is missing one or both files. Use the --new flag to create a new document.")
    NotepadApp(path, verbose=args.verbose, fresh=fresh).run()