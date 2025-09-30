import os
from urllib.error import URLError
import streamlit as st
from langchain_core.messages import HumanMessage

from src.components.runner import ChainRunner

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

model = "gemini-2.5-flash"
model_provider = "google_genai"
CONFIG_PATH = "./configs/inference_pipeline_config.yaml"

try:
    st.header("PDF RAG")
    st.subheader("a RAG to answer any questions from your given pdf document")
    RAG = ChainRunner(config_path=CONFIG_PATH).rag_chain

    if "model" not in st.session_state:
        st.session_state["model"] = model

    if "model_provider" not in st.session_state:
        st.session_state["model_provider"] = model_provider

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history and rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        print(f"prompt:\n{prompt}")

        with st.chat_message("assistant"):
            message_input = st.session_state.messages[-1]["content"]
            async def get_stream_response():
                """
                this function is used to extract content from the async_generator produced by
                RAG.astream()
                """
                stream = RAG.astream(message_input)
                async for chunk in stream:
                    try:
                        yield chunk["result"].replace("$","\$")
                    except:
                        pass

            print(f"write stream")
            response = st.write_stream(get_stream_response())
            print(f"response\n{response}")

        st.session_state.messages.append({"role": "assistant", "content": response.replace("$","\$")})

    if st.button("Reset chat"):
        st.session_state.clear()

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")