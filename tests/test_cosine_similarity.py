import unittest
import numpy as np
import sys
sys.path.append('../')
from scripts.cosine_similarity import calculate_cosine_distances

class TestCalculateCosineDistances(unittest.TestCase):

    def test_calculate_cosine_distances(self):
        sentences = [
            {'combined_sentence_embedding': np.array([1, 2, 3])},
            {'combined_sentence_embedding': np.array([2, 3, 4])},
            {'combined_sentence_embedding': np.array([3, 4, 5])}
        ]

        expected_distances = [0.0022689286287753465, 0.0022689286287753465]

        distances, sentences_with_distances = calculate_cosine_distances(sentences)
        self.assertEqual(distances, expected_distances)
        # You can also add more assertions to check sentences_with_distances if needed

if __name__ == '__main__':
    unittest.main()
