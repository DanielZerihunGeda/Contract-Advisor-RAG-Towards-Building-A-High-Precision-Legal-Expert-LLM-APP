import streamlit as st

# Main function to run the Streamlit app
def main():
  with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    st.title("Contract Advisory Chatbot")
    if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

        
    # Chat interaction section
    st.chat_message("user").write("Test")
    st.chat_message("assistant").write("Flwos")

    # File upload section (hidden by default)
    
if __name__ == "__main__":
    main()
