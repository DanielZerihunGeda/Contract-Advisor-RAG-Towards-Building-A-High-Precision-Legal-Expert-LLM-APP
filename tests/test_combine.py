import unittest
import sys
sys.path.append('../')
from scripts.combine import combine_sentences

class TestCombineSentences(unittest.TestCase):

    def test_combine_sentences(self):
        sentences = ['This is sentence 1.', 'This is sentence 2.', 'This is sentence 3.', 'This is sentence 4.', 'This is sentence 5.']
        expected_output = [
            {'sentence': 'This is sentence 1.',
 'index': 0,
 'combined_sentence': 'This is sentence 1. This is sentence 2. This is sentence 3.'},
            {'sentence': 'This is sentence 2.',
 'index': 1,
 'combined_sentence': 'This is sentence 1. This is sentence 2. This is sentence 3. This is sentence 4.'},
            {'sentence': 'This is sentence 3.',
 'index': 2,
 'combined_sentence': 'This is sentence 1. This is sentence 2. This is sentence 3. This is sentence 4. This is sentence 5.'},
            {'sentence': 'This is sentence 4.',
 'index': 3,
 'combined_sentence': 'This is sentence 2. This is sentence 3. This is sentence 4. This is sentence 5.'},
            {'sentence': 'This is sentence 5.',
 'index': 4,
 'combined_sentence': 'This is sentence 3. This is sentence 4. This is sentence 5.'}
        ]

        output = combine_sentences(sentences)
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()
