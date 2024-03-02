import numpy as np

def group_sentences_by_distance_threshold(distances, sentences):
    def get_sentence_text(sentence):
        try:
            return sentence['sentence']
        except Exception:
            raise ValueError("sentences list should contain dictionaries with a 'sentence' key.")

    breakpoint_percentile_threshold = 50
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    start_index = 0
    chunks = []

    for index in indices_above_thresh:
        end_index = index
        group = [get_sentence_text(sentence).strip() for sentence in sentences[start_index:end_index+1]]
        combined_text = ' '.join(group)
        chunks.append(combined_text)
        start_index = index + 1

    if start_index < len(sentences):
        end_index = len(sentences)
        group = [get_sentence_text(sentence).strip() for sentence in sentences[start_index:end_index]]
        combined_text = ' '.join(group)
        chunks.append(combined_text)
        chunks = [chunk for chunk in chunks if chunk]
    # Print the number of chunks
    print("Number of chunks:", len(chunks))

    # Print the text of each chunk
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print()

    return chunks