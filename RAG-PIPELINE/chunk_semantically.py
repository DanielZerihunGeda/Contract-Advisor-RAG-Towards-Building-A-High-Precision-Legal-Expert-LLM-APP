import re
import numpy as np
from combine_sentences import combine_sentences
import calculate_cosine_distance
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from calculate_cosine_distance import calculate_cosine_distances
def semantic_retriever(file):
    try:
        with open(file) as file:
            essay = file.read()
    except FileNotFoundError as e:
        print(f"Error: File '{file}' not found. {e}")
        return []

    try:
        single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
        
        try:
            sentences = combine_sentences(sentences)
        except Exception as e:
            print(f"Error in combine_sentences: {e}")
            return []

        try:
            oaiembeds = OpenAIEmbeddings()
            embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])
        except Exception as e:
            print(f"Error in OpenAI embeddings: {e}")
            return []

        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        try:
            distances, sentences = calculate_cosine_distances(sentences)
        except Exception as e:
            print(f"Error in calculate_cosine_distance: {e}")
            return []

        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
