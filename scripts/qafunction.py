from langchain.schema import format_document
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from scripts.vector_store import process_file_and_generate_vectorstore

def create_conversational_qa_chain(file_path):
    retriever = process_file_and_generate_vectorstore(file_path)
    
    template = """<human>: As a legal contract advisor, your role is pivotal in navigating the complexities of the contract clause agreement. Your responsibility extends beyond mere interpretation; it encompasses a meticulous analysis and strategic elucidation of the user's query based on the given {context}. To accomplish this effectively, consider the following steps:

    1. Contextual Understanding: Begin by immersing yourself in the provided contract clause or {context}, grasping the intricacies of the terms, conditions, and potential implications involved. Pay particular attention to any ambiguous language or clauses that may pose risks to your client.

    2. User Query Analysis: Carefully dissect the user's question within the {context} or the contract clause. Identify any specific clauses or provisions that the question pertains to and analyze their impact on your client's interests or question.

    3. Comprehensive Response: Craft a response that addresses the user's query comprehensively, taking into account the nuances of the contract clause. Provide detailed explanations and recommendations, ensuring that your client is fully informed of their rights and obligations under the agreement.

    4. Risk Mitigation: As you formulate your response, prioritize the mitigation of risks inherent in the contract clause. Identify potential pitfalls or areas of concern and propose proactive measures to safeguard your client's interests and minimize their exposure to liability.

    5. Strategic Communication: Ensure that your response is conveyed in a clear, concise, and persuasive manner. Articulate your points effectively, citing relevant legal precedents, industry standards, and best practices to support your arguments and strengthen your client's position.

    By following these steps diligently, you can provide informed, strategic guidance that empowers your client and ensures the integrity of the contract clause agreement.

    NB: If the provided {context} is not sufficient enough to answer the user question please avoid any assumption.
    Question: {question}


    \n

    <bot>:
    """

    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    
    return conversational_qa_chain
