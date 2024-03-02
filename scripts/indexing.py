from dotenv import load_dotenv
import os
load_dotenv
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def local_index(chunks, file_path):
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    vectorstore = FAISS.from_texts(
        chunks, embedding=embeddings
    )
    vectorstore.save_local(file_path)