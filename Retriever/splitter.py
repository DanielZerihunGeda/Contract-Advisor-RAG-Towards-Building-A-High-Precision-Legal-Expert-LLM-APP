import docx
import spacy

def extract_sentences_from_docx(docx_filepath):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # Load the Word document
    doc = docx.Document(docx_filepath)

    # Extract text from paragraphs
    essay = ""
    for paragraph in doc.paragraphs:
        essay += paragraph.text + "\n"

    # Process the essay text
    doc = nlp(essay)

    # Extract individual sentences
    single_sentences_list = [sent.text for sent in doc.sents]

    return single_sentences_list