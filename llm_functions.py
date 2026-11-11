# Load variables from the .env file into the environment
import os
from dotenv import load_dotenv
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from wordllm_doc import WordLLMDocument, Block, Run, load_document, save_document, document_to_text, text_to_blocks

env=load_dotenv(dotenv_path=".env", override=True)

if not env:
    with open(".env",'w') as config:
        config.write("SPREADSHEET_PATH=\"spreadsheet.json\"")

gemini_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

counters = {"search":0}

@tool
def search_tool(query: str,max_results:int) -> str:
    """Tavily web search query information. Use less max_results for simple
    factual lookups and more max_results for broad research."""
    counters["search"] += 1
    if counters["search"] > 5:
        return "Search limit reached. Answer with what you already have."
    try:
        search = TavilySearch(max_results=max_results, api_key=tavily_key)
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Error during search: {str(e)}"


class WordProcessorAssistant():
    def __init__(self, model_name:str = "gemini-3.1-flash-lite", document_path:str = "draft.json", verbose:bool = False):
        self.agent =create_agent(model=ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key),
                                 tools=[search_tool],
                                 system_prompt="You are secretary. You do user request with document. Learn document tone and style in manifest.")
        self.document = load_document(document_path)
        self.manifest = self.document.manifest
        self.verbose = verbose

    def answer_question(self, question:str, highlight:Optional[str] = None) -> str:
        counters["search"] = 0  # Reset search counter for each question
        prompt = f"Document manifest: {self.manifest}. Document content: {document_to_text(self.document)}. Question: {question}"
        if highlight:
            prompt += f" Focus on this highlighted text: {highlight}."
        result = self.agent.invoke({"messages":[{"role":"system","content":"You secretary. Use search_tool answer document question. If user highlight, focus highlighted text."},
                                              {"role":"user","content":prompt}]})
        if self.verbose:
            #return str(result)
            return("\n\n".join(str(m.type)+":\n"+str(m.content) for m in result["messages"]))
        return str(result["messages"][-1].content[-1]["text"])