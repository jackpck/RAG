import os
from urllib.error import URLError
import streamlit as st
from langchain_core.messages import HumanMessage
import datetime

from src.components.runner import ChainRunner
from src.utils.sql_connection import NeonPostgres
from credentials import db_config

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

model = "gemini-2.5-flash"
model_provider = "google_genai"
CONFIG_PATH = "./configs/inference_pipeline_config.yaml"
tablename = "chatbot_log"

def add_feedback():
    feedback = st.session_state.feedback_value
    neon_db.cur.execute(f"""
        UPDATE {tablename}
        SET user_feedback = {feedback}
        WHERE id = (SELECT MAX(id) FROM {tablename})
    """)
    neon_db.commit()

try:
    st.header("PDF RAG")
    st.subheader("a RAG to answer any questions from your given pdf document")

    RAG = ChainRunner(config_path=CONFIG_PATH).rag_chain

    neon_db = NeonPostgres(dbname=db_config.db_name,
                           user=db_config.db_user,
                           password=db_config.db_pwd,
                           host=db_config.db_host,
                           port=db_config.db_port)

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
        print("wait user input")
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
                    if "result" in chunk:
                        try:
                            result = chunk['result'].content.replace('$','\$')
                            yield f"\n\n Response \n: {result}"
                        except:
                            pass

                    if "citation" in chunk:
                        citations = chunk["citation"]
                        print(f"citations: {citations}")
                        if isinstance(citations, list):
                            citation_source = [cite.page_content for cite in citations]
                            yield f"\n\n Citations: " + "\n-".join(citation_source)
                        else:
                            yield f"\n\n Citations: \n-{citations.page_content}"

            response = st.write_stream(get_stream_response())

            # add feedback feature
            feedback = st.feedback(
                "thumbs",
                key=f"feedback_value",
                on_change=add_feedback,
            )

            st.session_state.messages.append({"role": "assistant",
                                              "content": response.replace("$", "\$"),
                                              "feedback": feedback})

            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            neon_db.create_log_table(tablename=tablename)
            neon_db.insert_feedback_to_table(tablename=tablename,
                                             datetime=datetime_str,
                                             user_query=prompt,
                                             response=response,
                                             user_feedback=feedback)
            neon_db.commit()

    if st.button("Reset chat"):
        st.session_state.clear()

except URLError as e:
    st.error(f"This demo requires internet access. Connection error: {e.reason}")