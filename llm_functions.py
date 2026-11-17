# Load variables from the .env file into the environment
import os
from dotenv import load_dotenv
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_core.documents import Document as LangchainDocument
from langchain_core.tools import StructuredTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
import numpy as np

from wordllm_doc import WordLLMDocument, load_document
from retriever import Retriever

MAX_CONTEXT_WINDOW = 4096
APPROX_CHARS_PER_TOKEN = 8

env=load_dotenv(dotenv_path=".env", override=True)

if not env:
    with open(".env",'w') as config:
        config.write("#Fill keys as needed\\" \
        "#GOOGLE_API_KEY=\\"
        "#TAVILY_API_KEY=")

gemini_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

counters = {"search":0,"retrieve":0}
tavily = TavilySearch(api_key=tavily_key)

def chunk_text(text:str, chunk_size:int=4000, chunk_overlap:int=200) -> list[LangchainDocument]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = [LangchainDocument(page_content=t) for t in splitter.split_text(text)]
    return docs

def capped_search(**kwargs) -> str:
    counters["search"] += 1
    if counters["search"] > 5:
        return "Search budget reached. Answer with what you already have."
    try:
        results = tavily.invoke(kwargs)
        return str(results)
    except Exception as e:
        return f"Error during search: {str(e)}"
search_tool = StructuredTool.from_function(
    func=capped_search,
    name=tavily.name,
    description=tavily.description,
    args_schema=tavily.args_schema,
)

class WordProcessorAssistant():
    def __init__(self, model_name:str = "gemini-3.5-flash-lite", document_path:str = "draft.json", verbose:bool = 0, fresh:bool = False) -> None:
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key)
        self.doc_path = document_path
        if fresh:
            self.doc = WordLLMDocument(title="New Document", manifest="A general purpose document", blocks=dict())
            self.doc.save(self.doc_path)
        else:
            self.doc = load_document(document_path)
        self.manifest = self.doc.manifest
        self.verbose = verbose
        self.retriever = Retriever(self.doc_path,self.doc)

    def save_doc_and_retriever(self,text):
        self.doc.update_from_text(text)
        synced_blocks = self.retriever.sync(self.doc)
        self.retriever._save()
        self.doc.save(self.doc_path)
        return synced_blocks


    def summarize_doc(self, document: WordLLMDocument, context_window_size: int) -> str:
        try:
            text = document.get_text()
            if len(text) > context_window_size:
                #If text is long, use map-reduce technique to summarize in chunks
                docs = chunk_text(text, chunk_size=context_window_size, chunk_overlap=200)
                return load_summarize_chain(self.llm, chain_type="map_reduce").invoke({"input_documents": docs})["output_text"]
            #Otherwise, just use the whole text
            return text
        except Exception as e:
            return f"Error during summarization: {str(e)}"

    def retrieve(self, query:str):
        if counters['retrieve'] < 1:
            blocks = list(self.doc.blocks.values())
            scores = self.retriever.search(query)        
            top = np.argsort(-scores)
            response = ""
            response_len = 0
            chunk_num=0
            for index in top:
                lo = max(0,index-1)
                hi = min(index+1,len(blocks)-1)
                needed_indices = [index]
                if lo < index:
                    needed_indices = [lo,index]
                if index < hi:
                    needed_indices.append(hi)
                neighbor_blocks = [blocks[i] for i in needed_indices]
                text = "\n".join([f"Block ID:{b.id}, Chunk Text:{b.text}" for b in neighbor_blocks])
                text = f"Chunk {chunk_num}("+text+f") Similarity:{float(scores[index])} \n"
                text_len = len(text)
                if text_len + response_len < MAX_CONTEXT_WINDOW:
                    response += text
                    response_len += text_len
                else:
                    break
                chunk_num += 1
        
            counters['retrieve']+=1
            return response
        else:
            return "Retrieval budget reached. Respond with what you already have."
        
    def reset_counters(self):
        for v in counters.values():
            counters[v] = 0
        if self.verbose>0:
            print(counters)

    def modify_manifest(self, new_manifest:str) -> str:
        self.doc.manifest = new_manifest
        self.doc.save(self.doc_path)
        return f"Manifest updated to: {new_manifest}"

    def rename_document(self, new_title:str) -> str:
        self.doc.title = new_title
        self.doc.save(self.doc_path)
        return f"Document title updated to: {new_title}"
    

    def answer_question(self, question:str, highlight:Optional[str] = None) -> str:
        counters["search"] = 0  # Reset search counter for each question
        counters["retrieve"]=0
        prompt = f"Document manifest: {self.manifest}\n Question: {question}"
        if highlight:
            prompt += f"\nFocus on this highlighted text: {highlight}."

        modify_manifest_tool = StructuredTool.from_function(self.modify_manifest, name="modify_manifest", description="Changes the document manifest to match the new_manifest string. The new_manifest describes a change in the intended tone, style, and 'vibes' of the document, not the factual details or events. Use this tool if and only if the user explictly requests such a change. It does not change the document text, and should not be used as such.")
        rename_document_tool = StructuredTool.from_function(self.rename_document,name="rename_document",description="Changes the internal title of the document to match the new_title string. Note that this does not change the file name in memory.")
        retriever_tool = StructuredTool.from_function(self.retrieve,name="retrieve",description="Does a fuzzy semantic search for information pertaining to 'query', returning chunks of text surrounding the semantically similar block up to a length set in the configuration.")

        agent = create_agent(self.llm, tools=[search_tool,modify_manifest_tool,rename_document_tool,retriever_tool], system_prompt="You are a secretary that can read, interact with, and answer questions about a specific document. Combine information from the retriever_tool and search_tool to do what the user requests, and respond with a string. A user may highlight a specific section of the document to focus on. If the task is not possible with the document or any tools, respond saying so.")

        result = agent.invoke({"messages":[{"role":"user","content":prompt}]})
        if self.verbose > 0:
            return("\n\n".join(str(m.type)+":\n"+str(m.content[:self.verbose]) for m in result["messages"]))
        if isinstance(content:=result["messages"][-1].content,str):
            return content
        else: 
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            return "".join(parts)