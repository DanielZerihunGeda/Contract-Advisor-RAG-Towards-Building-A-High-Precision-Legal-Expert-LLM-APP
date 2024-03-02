from scripts.splitter import split_into_sentences_from_file 
from scripts.chunk import group_sentences_by_distance_threshold
from scripts.cosine_similarity import calculate_cosine_distances
from scripts.embedding import get_sentence_embeddings
from scripts.indexing import local_index
from scripts.combine import combine_sentences
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import os
load_dotenv

def process_file_and_generate_vectorstore(file_path):
    # Split the file into individual sentences
    single_sentence = split_into_sentences_from_file(file_path)

    # Combine the sentences into a coherent form
    sentences = combine_sentences(single_sentence)

    # Generate embeddings for each sentence
    embeddings = get_sentence_embeddings(sentences)

    # Calculate cosine distances between embeddings
    distances, sentences = calculate_cosine_distances(embeddings)

    # Group sentences based on distance threshold
    chunks = group_sentences_by_distance_threshold(distances, sentences)

    # Initialize Spacy embeddings
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

    # Create a FAISS vector store from the chunks of text
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k" : 20}) 
    compressor = CohereRerank()

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever)
    return compression_retriever