import os
from urllib.error import URLError
import streamlit as st
from langchain_core.messages import HumanMessage

from src.components.runner import ChainRunner

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

model = "gemini-2.5-flash"
model_provider = "google_genai"
CONFIG_PATH = "./configs/inference_pipeline_config.yaml"

try:
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

        with st.chat_message("assistant"):
            message_input = [HumanMessage(content=st.session_state.messages[-1]["content"])]
            async def get_stream_response():
                """
                this function is used to extract content from the async_generator produced by
                RAG.astream()
                """
                stream = RAG.astream({"messages": message_input})
                async for chunk in stream:
                    try:
                        yield chunk.content.replace("$","\$")
                    except:
                        pass

            response = st.write_stream(get_stream_response())

        st.session_state.messages.append({"role": "assistant", "content": response.replace("$","\$")})

    if st.button("Reset chat"):
        st.session_state.clear()

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")