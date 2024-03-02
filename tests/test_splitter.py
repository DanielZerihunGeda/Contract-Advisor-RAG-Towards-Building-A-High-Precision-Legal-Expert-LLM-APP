import unittest
from unittest.mock import patch
from io import StringIO
import sys
sys.path.append('../')
from scripts.splitter import split_into_sentences_from_file

class TestSplitIntoSentencesFromFile(unittest.TestCase):

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='<html><body>This is a test. This is another test!</body></html>')
    def test_html_file(self, mock_open):
        sentences = split_into_sentences_from_file('test.html')
        self.assertEqual(sentences, ['This is a test.', 'This is another test!'])

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='This is a test. This is another test!')
    def test_text_file(self, mock_open):
        sentences = split_into_sentences_from_file('test.txt')
        self.assertEqual(sentences, ['This is a test.', 'This is another test!'])

    # Add similar test cases for PDF and DOCX files

    @patch('requests.get')
    def test_html_url(self, mock_get):
        mock_response = unittest.mock.Mock()
        mock_response.text = '<html><body>This is a test. This is another test!</body></html>'
        mock_response.headers.get.return_value = 'text/html'
        mock_get.return_value = mock_response

        sentences = split_into_sentences_from_file('http://example.com/test.html')
        self.assertEqual(sentences, ['This is a test.', 'This is another test!'])

    # Add similar test cases for PDF URLs

if __name__ == '__main__':
    unittest.main()
