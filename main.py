from scripts.qafunction import create_conversational_qa_chain
conversational_qa_chain = create_conversational_qa_chain("data/Robinson Advisory.docx")
example_message = conversational_qa_chain.invoke(
    {
        "question": "what is the contract about?",
        "chat_history": [],
    }
)