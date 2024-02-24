from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from Retriever.agentic_chunker import AgenticChunker

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    ac = AgenticChunker()
    essay_propositions = []

    for i, para in enumerate(paragraphs[:5]):
        propositions = get_propositions(para)
        
        essay_propositions.extend(propositions)
        print (f"Done with {i}")
    def __init__(self):
        self.model = ChatOpenAI()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
           """<human>: you are rag pipeline design consultant with deepest understanding on the technical nuances of RAG system your job is to provide code as well as deep explanation from question answer the question:

                        ### CONTEXT
                        {context}

                        ### QUESTION
                        Question: {question}

                        \n

                        <bot>:
                        """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None