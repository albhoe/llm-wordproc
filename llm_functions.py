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
from llm_proc_doc import LLMProcDocument, Block, Run, load_document, save_document, document_to_text, text_to_blocks
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

env=load_dotenv(dotenv_path=".env", override=True)

if not env:
    with open(".env",'w') as config:
        config.write("SPREADSHEET_PATH=\"spreadsheet.json\"")

gemini_key = os.getenv("GOOGLE_API_KEY")

# Use your api_key normally below
print(f"Loaded key successfully: {gemini_key[:4]}...") 


class WordProcessorAssistant():
    def __init__(self, model_name:str = "gemini-3.1-flash-lite", document_path:str = "draft.json"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key)
        self.document = load_document(document_path)
        self.manifest = self.document.manifest

    def answer_question(self, question:str, highlight:Optional[str] = None) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system","You are a writing assistant that can answer questions about a document. A user may optionally provide you with context for the question."),
            ("user", "The document is about this: {manifest}"),
            ("user", "Document context:{context}"),
            ("user", "Question: {question}")])
        
        chain = create_stuff_documents_chain(self.llm, prompt)
        if highlight:
            context = [LCDocument(page_content=highlight)]
        else:
            context = [LCDocument(page_content=document_to_text(self.document))]
        return chain.invoke({"context": context, "manifest": self.manifest, "question": question})