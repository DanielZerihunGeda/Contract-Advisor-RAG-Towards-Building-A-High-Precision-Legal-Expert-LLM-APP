import unittest
import sys
sys.path.append('../')
from scripts.embedding import get_sentence_embeddings

class TestGetSentenceEmbeddings(unittest.TestCase):

    def test_get_sentence_embeddings(self):
        sentences = [
            {'combined_sentence': 'This is sentence 1.'},
            {'combined_sentence': 'This is sentence 2.'},
            {'combined_sentence': 'This is sentence 3.'}
        ]

        sentences_with_embeddings = get_sentence_embeddings(sentences)

        # Check if each sentence now has an embedding vector
        for sentence in sentences_with_embeddings:
            self.assertIn('combined_sentence_embedding', sentence)
            self.assertIsNotNone(sentence['combined_sentence_embedding'])
            self.assertIsInstance(sentence['combined_sentence_embedding'], list)
            self.assertNotEqual(len(sentence['combined_sentence_embedding']), 0)

if __name__ == '__main__':
    unittest.main()
