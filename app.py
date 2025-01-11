import streamlit as st

# Main function to run the Streamlit app
def main():
    st.title("Contract Advisory Chatbot")
    
    # Chat interaction section
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        uploaded_file = st.session_state.get("uploaded_file")
        if uploaded_file is None:
            st.text("Please upload a document file before asking questions.")
        else:
            conversational_qa_chain = create_conversational_qa_chain_with_file(uploaded_file)
            if conversational_qa_chain:
                example_message = conversational_qa_chain.invoke(
                    {
                        "question": user_question,
                        "chat_history": [],  # For simplicity, we initialize an empty chat history
                    }
                )
                # Display chat history
                for message in example_message["chat_history"]:
                    if message.sender == "human":
                        st.text(f"You: {message.content}")
                    elif message.sender == "bot":
                        st.text(f"Bot: {message.content}")

    # File upload section (hidden by default)
    with st.expander("Upload Document File", expanded=False):
        uploaded_file = st.file_uploader("", type=[".docx", ".pdf", ".txt"])
        st.session_state["uploaded_file"] = uploaded_file

if __name__ == "__main__":
    main()
