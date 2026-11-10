import anthropic

client = anthropic.Anthropic()

# in NotepadApp:
BINDINGS = [
    ...
    Binding("ctrl+l", "ask_llm", "Ask AI"),
]

async def _call_llm(self, selected: str) -> None:
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-5-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"Rewrite this text more concisely:\n\n{selected}",
        }],
    )
    reply = response.content[0].text
    self.notify("LLM responded")
    self.last_reply = reply