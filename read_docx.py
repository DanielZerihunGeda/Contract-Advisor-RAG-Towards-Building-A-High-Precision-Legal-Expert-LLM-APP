from docx import Document

def read_docx(file_path):
    """
    Read a .docx file and return its text content as a string.

    Args:
    - file_path (str): The path to the .docx file.

    Returns:
    - text (str): The text content of the .docx file.
    """
    try:
        # Load the .docx file
        doc = Document(file_path)

        # Initialize an empty string to store the text content
        text = ""

        # Iterate through each paragraph in the document and extract text
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text
    except Exception as e:
        print(f"Error reading .docx file: {e}")
        return ""
