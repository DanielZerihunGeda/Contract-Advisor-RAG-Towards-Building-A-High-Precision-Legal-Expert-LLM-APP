import spacy

# Load the spaCy embedder



def get_sentence_embeddings(sentences):
    nlp = spacy.load('en_core_web_sm')

# Create embedding vectors for the input sentences using the spaCy embedder
    embeddings = [nlp(sentence).vector for sentence in [x['combined_sentence'] for x in sentences]]
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    return sentences