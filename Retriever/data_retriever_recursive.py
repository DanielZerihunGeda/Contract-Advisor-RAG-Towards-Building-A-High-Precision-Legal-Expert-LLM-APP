from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import combine_sentences
import re
def data_retriever(file, query):
    
    # Load the document, split it into chunks, embed each chunk, and load it into the vector store.
    raw_documents = TextLoader(file).load()

    """
    splitting the text recursively for since it is the recommended one for generic text.
    It tries to split on them in order until the chunks are small enough
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
    documents = text_splitter.split_documents(raw_documents)
    
    # Assuming OpenAIEmbeddings is properly initialized and available as `embeddings`
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    
    query = query

    """
    Assigning score_threshold to 0.5 to retrieve document with above 
    50% of relevance score, and setting "K" == 5 to retrieve five chunks
    for a single query
    """
    retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.65, "k": 5}
)

    docs = retriever.get_relevant_documents(query)
    
    # Print each output individually
    for i in range(min(4, len(docs))):  # Avoid index out of range if there are less than 4 results
        print(docs[i].page_content)
    
    return None
