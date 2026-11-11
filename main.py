from pathlib import Path

from llm_proc_doc import Document, save_document, load_document
from render import render_document

FILE = Path("draft.json")


def main() -> None:
    if FILE.exists():
        doc = load_document(FILE)
    else:
        doc = Document(title="My First Draft")
        doc.add_paragraph("Introduction", kind="heading", level=1)
        doc.add_paragraph("This is the first paragraph of the document.")
        doc.add_paragraph("Second point, as a list item.", kind="list_item")
        save_document(doc, FILE)

    render_document(doc)


if __name__ == "__main__":
    main()