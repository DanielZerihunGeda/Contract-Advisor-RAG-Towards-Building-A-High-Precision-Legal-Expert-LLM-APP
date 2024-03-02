import unittest
import numpy as np
import sys
sys.path.append('../')
from scripts.chunk import group_sentences_by_distance_threshold

class TestGroupSentencesByDistanceThreshold(unittest.TestCase):

    def test_group_sentences(self):
        distances = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        sentences = [{'sentence': 'This is sentence 1.'},
                     {'sentence': 'This is sentence 2.'},
                     {'sentence': 'This is sentence 3.'},
                     {'sentence': 'This is sentence 4.'},
                     {'sentence': 'This is sentence 5.'},
                     {'sentence': 'This is sentence 6.'},
                     {'sentence': 'This is sentence 7.'},
                     {'sentence': 'This is sentence 8.'},
                     {'sentence': 'This is sentence 9.'}]

        expected_chunks = [
            'This is sentence 1. This is sentence 2. This is sentence 3. This is sentence 4.',
            'This is sentence 5. This is sentence 6. This is sentence 7. This is sentence 8. This is sentence 9.'
        ]

        chunks = group_sentences_by_distance_threshold(distances, sentences)
        self.assertEqual(chunks, expected_chunks)

if __name__ == '__main__':
    unittest.main()
