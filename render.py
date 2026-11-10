from rich.console import Console
from rich.table import Table
from rich.text import Text

from document import Block, Document

console = Console()


def run_to_text(block: Block) -> Text:
    """Build a rich Text object so bold/italic survive into the terminal."""
    t = Text()
    for run in block.runs:
        style = " ".join(s for s, on in (("bold", run.bold),
                                         ("italic", run.italic)) if on)
        t.append(run.text, style=style or None)
    return t


def render_block(block: Block) -> Text:
    body = run_to_text(block)
    if block.kind == "heading":
        body.stylize("bold underline")
        body = Text(f"{'#' * max(block.level, 1)} ") + body
    elif block.kind == "list_item":
        indent = "  " * block.level
        body = Text(f"{indent}• ") + body
    return body


def render_document(doc: Document) -> None:
    console.rule(f"[bold]{doc.title}[/bold]")

    # One row per block: a dim ID column, then the content.
    table = Table(show_header=False, box=None, pad_edge=False, expand=True)
    table.add_column("id", style="dim cyan", no_wrap=True, width=10)
    table.add_column("content", ratio=1)

    for block in doc.blocks:
        table.add_row(f"[{block.id}]", render_block(block))

    console.print(table)
    console.rule()


def render_plain(doc: Document) -> str:
    """Plain-text version, useful for sending to the LLM as context."""
    lines = []
    for b in doc.blocks:
        lines.append(f"[{b.id}] ({b.kind}) {b.text}")
    return "\n".join(lines)