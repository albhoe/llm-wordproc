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
from fuzzysearch import FuzzySearchIterator
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

counters = {"search":5,"retrieve":5,"edit":5}
tavily = TavilySearch(api_key=tavily_key)

def chunk_text(text:str, chunk_size:int=4000, chunk_overlap:int=200) -> list[LangchainDocument]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = [LangchainDocument(page_content=t) for t in splitter.split_text(text)]
    return docs

def capped_search(**kwargs) -> str:
    counters["search"] -= 1
    if counters["search"] < 0:
        return "Search budget expended. Answer with what you already have."
    try:
        results = tavily.invoke(kwargs)
        return str(results)
    except Exception as e:
        return f"Error during search: {e!r}"
search_tool = StructuredTool.from_function(
    func=capped_search,
    name=tavily.name,
    description=tavily.description,
    args_schema=tavily.args_schema,
)

def content_text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    # list of content blocks, e.g. [{"type": "text", "text": "..."}]
    return "".join(
        b["text"] for b in c
        if isinstance(b, dict) and b.get("type") == "text"
    )

class WordProcessorAssistant():
    def __init__(self, document_path, model_name:str = "gemini-3.5-flash-lite", verbose:bool = 0, fresh:bool = False) -> None:
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key)
        self.doc_path = document_path
        if fresh:
            self.doc = WordLLMDocument(title="New Document", manifest="A general purpose document", blocks=dict())
        else:
            try:
               self.doc = load_document(document_path)
            except Exception as e:
                raise FileNotFoundError(f"Error while trying to load document {document_path}, {e!r}")
        self.retriever = Retriever(self.doc_path,self.doc)
        self.manifest = self.doc.manifest
        self.verbose = verbose
        self.fsiterator:FuzzySearchIterator | None = None
        self.selection: tuple[tuple[int,int],tuple[int,int]] = ((0,0),(0,0))
        self.save_doc_and_retriever(self.doc.get_text())

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
            return f"Error during summarization: {e!r}"

    def reset_counters(self):
        global counters 
        counters = {"search":5,"retrieve":5,"edit":5}
        if self.verbose:
            print(counters)

    def retrieve(self, query:str):
        try:
            counters['retrieve']-=1
            if counters['retrieve'] >= 0:
                scores = self.retriever.search(query)        
                top = np.argsort(-scores)
                response = ""
                response_len = 0
                chunk_num=0
                for index in top:
                    lo = max(0,index-1)
                    hi = min(index+1,len(self.doc.blocks)-1)
                    needed_indices = [index]
                    if lo < index:
                        needed_indices = [lo,index]
                    if index < hi:
                        needed_indices.append(hi)
                    neighbor_blocks = [self.doc.blocks[i] for i in needed_indices]
                    text = "\n".join([f"(Block ID:{b.id}, Block Text:\"{b.text}\n\")," for b in neighbor_blocks])
                    text = f"Chunk {chunk_num} similarity {float(scores[index])}: ["+text+f"],"
                    text_len = len(text)
                    if text_len + response_len < MAX_CONTEXT_WINDOW:
                        response += text
                        response_len += text_len
                    else:
                        break
                    chunk_num += 1
            
                if self.verbose: response = "query:{query}, response:"+response
                return response
            else:
                return "Retrieval budget reached. Respond with what you already have."
        except Exception as e:
            return f"Error retrieving context: {e!r}"



    def modify_manifest(self, new_manifest:str) -> str:
        self.doc.manifest = new_manifest
        self.doc.save(self.doc_path)
        return f"Manifest updated"

    def rename_document(self, new_title:str) -> str:
        self.doc.title = new_title
        self.doc.save(self.doc_path)
        return f"Document title updated"

    def create_fsiterator(self, block_ids:list[str])-> str:
        try:
            nonmatch = set(block_ids)-set([b.id for b in self.doc.blocks])
            if len(nonmatch)==0:
                self.fsiterator = FuzzySearchIterator(block_ids)
                res = "Iterator created successfully."
                return res
            else:
                return f"{nonmatch} was not in the document block list. Double check it."
        except Exception as e:
            return f"Error creating fsiterator {e!r}"

    def destroy_fsiterator(self):
        self.fsiterator = None
        return

    def fsiterator_next(self) -> Optional[int]:
        """Outputs the line number of the next block_id in the iterator"""
        if self.fsiterator:
            id = self.fsiterator.next()
            try:
                return self.doc.index[id]
            except:
                self.doc.reindex()
                return self.doc.index[id]
        else: 
            return None

    def fsiterator_prev(self) -> Optional[int]:
        """Outputs the line number of the previous block_id in the iterator"""
        if self.fsiterator:
            id = self.fsiterator.prev()
            try:
                return self.doc.index[id]
            except:
                self.doc.reindex()
                return self.doc.index[id]
    
    def answer_question(self, question:str, selected:Optional[str] = None, selected_location:tuple[tuple[int,int],tuple[int,int]]=((0,0),(0,0))) -> str:
        self.reset_counters()
        ((r1,c1),(r2,c2)) = selected_location
        r1 = r1//2
        r2 = r2//2
        selected_block_ids = [self.doc.blocks[ri].id for ri in range(r1,r2+1)]
        prompt = f"Document title: {self.doc.title} Document manifest: {self.manifest}\n Question: {question}"
        if selected:
            prompt += f"\nFocus on this selected text: \"{selected}\" in blocks [{selected_block_ids}]."

        modify_manifest_tool = StructuredTool.from_function(self.modify_manifest, name="modify_manifest", description="Changes the document manifest to match the new_manifest string. The new_manifest describes a change in the intended tone, style, and 'vibes' of the document, not the factual details or events. Use this tool if and only if the user explictly requests such a change. It does not change the document text, and should not be used as such.")
        rename_document_tool = StructuredTool.from_function(self.rename_document,name="rename_document",description="Changes the internal title of the document to match the new_title string. Note that this does not change the file name in memory.")
        retriever_tool = StructuredTool.from_function(self.retrieve,name="retrieve",description="Does a fuzzy semantic search for information pertaining to 'query', returning chunks of text surrounding the semantically similar block up to a length set in the configuration. For unknown names and words, consider followup questions to retrieve or tavily_search")
        create_fsiterator_tool = StructuredTool.from_function(self.create_fsiterator, name="create_fsiterator", description="Creates an iterator object that will be visible on the user interface. If a user asks for sections of text, put the appropriate block_ids into this function. Do not put strings other than valid block ids into the list.")
        
        blocks = self.doc.blocks
        assert (0<=r1<=r2<len(blocks)), "Assert the selection is valid for the document."
        WINDOW = 16
        startbuffer=blocks[r1].text[:c1]
        endbuffer=blocks[r2].text[c2:]
        tempstartbuffer=""
        tempendbuffer=""

        def edit_block(block_id:str,text_parameter:str) -> str:
            try:
                counters["edit"] -= 1
                if counters["edit"] >= 0:
                    ri = self.doc.index[block_id]
                    block = blocks[self.doc.index[block_id]]
                    if ri < r1 or r2 < ri:
                        return "This block is outside the user's selected range, and therefore should not be modified. Ask the user to confirm intent."
                    #If the block contains the endpoint of the selected range (ri==r1)|(ri==r2), enforce the constraint based on one or both columns. Otherwise, tool is free to modify everything.
                    if r1==ri:
                        tempstartbuffer = startbuffer
                    if ri==r2:
                        tempendbuffer = endbuffer
                    block.update_from_text(tempstartbuffer+text_parameter+tempendbuffer)
                    tempstartbuffer = tempstartbuffer[-min(WINDOW,len(tempstartbuffer)):]
                    tempendbuffer = tempendbuffer[:min(WINDOW,len(tempendbuffer))]
                    return f"The neighboring text is now: \"{tempstartbuffer}{text_parameter}{tempendbuffer}\" Verify that this has the correct spacing and punctuation before and after each word, but incomplete words at the beginning and end of the fragment are safe. If it is not correct, try again with different spacing and punctuation around text_parameter. Also try to keep asterisks for markup syntax if there were any in the original selected text."
                else:
                    return "Edit budget expended. Allow the user to intervene manually."
            except Exception as e:
                return f"Exception during edit: {e!r}"
        edit_block_tool = StructuredTool.from_function(edit_block,name="edit_block",description="Edits the text field of a particular block. This can only edit the text which the user has selected to focus on. It cannot edit any text outside the user's selection. Try to preserve the spaces, punctuation, and asterisks at the beginning and end of the original selected text when creating text_parameter, so the resulting text fragment reads cleanly.")


        agent = create_agent(self.llm, tools=[search_tool,modify_manifest_tool,rename_document_tool,retriever_tool,create_fsiterator_tool,edit_block_tool], system_prompt="You are a secretary that can read, interact with, and answer questions about a specific document. Combine information from retrieve and tavily_search to do what the user requests, and respond with a string. A user may select a specific section of the document to focus on. If the task is not possible with the document or any tools, respond saying so.")

        result = agent.invoke({"messages":[{"role":"user","content":prompt}]})
        if self.verbose > 0:
            parts = [str(counters)]
            for msg in result["messages"]:
                if msg.type == "ai" and msg.tool_calls:
                    for call in msg.tool_calls:
                        parts.append(f"**Tool call:** `{call['name']}`\n```json\n{call['args']}\n```")
                else:
                    text = content_text(msg)[: self.verbose]
                    parts.append(f"**{msg.type}:**\n{text}")
            return "\n---\n".join(parts)

        return content_text(result["messages"][-1])