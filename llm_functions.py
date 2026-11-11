# Load variables from the .env file into the environment
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import Tool, tool, StructuredTool
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_tavily import TavilySearch

from llm_proc_doc import LLMProcDocument, Block, Run, load_document, save_document, document_to_text, text_to_blocks

env=load_dotenv(dotenv_path=".env", override=True)

if not env:
    with open(".env",'w') as config:
        config.write("SPREADSHEET_PATH=\"spreadsheet.json\"")

gemini_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

search_counter = 0
@tool
def search_tool(query: str,max_results:int) -> str:
    """Tavily web search query information. Use less max_results for simple
    factual lookups and more max_results for broad research."""
    search_counter += 1
    if search_counter > 5:
        return "Search limit reached. Answer with what you already have."
    try:
        search = TavilySearch(max_results=max_results, api_key=tavily_key)
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Error during search: {str(e)}"


class WordProcessorAssistant():
    def __init__(self, model_name:str = "gemini-3.1-flash-lite", document_path:str = "draft.json"):
        self.agent =create_agent(model=ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key),
                                 tools=[search_tool],
                                 system_prompt="You are secretary. You access document and web search tool. Follow document manifest.")
        self.document = load_document(document_path)
        self.manifest = self.document.manifest
        self.verbose = False

    def answer_question(self, question:str, highlight:Optional[str] = None) -> str:
        prompt = f"Document manifest: {self.manifest}. Document content: {document_to_text(self.document)}. Highlighted text: {highlight}. Question: {question}"
        if highlight:
            prompt += f" Focus on the highlighted text: {highlight}."
        result = self.agent.invoke({"messages":[{"role":"system","content":"You secretary. Use search_tool answer document question. If user highlight, focus highlighted text."},
                                              {"role":"user","content":prompt}]})
        if self.verbose:
            return("\n\n".join(str(m.content) for m in result["messages"]))
        return str(result["messages"][-1].content[-1]["text"])