import pytest

from src.components.retriever import ChunkRetriever, PostgresRetriever
from src.components.embedder import DocEmbedder
import data.test.test_user_query_examples as test_examples

top_k_retrieved = 10

@pytest.fixture()
def my_vectorstore():
    model_name = "all-MiniLM-L6-v2"
    vs_name = "faiss_index_google_genai_risk_mgmt"
    embedder = DocEmbedder(model_name=model_name,
                           vs_name=vs_name)
    vectorstore = embedder.from_vs()
    return vectorstore

@pytest.fixture()
def my_retriever(my_vectorstore):
    params = {"retriever_search_type":"similarity",
              "retriever_search_kwargs": {"k":top_k_retrieved}}
    retriever = ChunkRetriever(**params)
    return retriever.retrieve(vectorstore=my_vectorstore)

@pytest.fixture()
def my_pg_retriever():
    tablename = "faiss_index_google_genai_risk_mgmt"
    embedding_model_name = "all-MiniLM-L6-v2"
    neon_db = PostgresRetriever(tablename=tablename,
                                embedding_model_name=embedding_model_name)
    return neon_db


def test_retrieval_k(my_retriever):
    retrieved_doc = my_retriever.invoke(test_examples.user_query_1)
    assert len(retrieved_doc) == top_k_retrieved

def test_retrieval_page(my_retriever):
    retrieved_doc = my_retriever.invoke(test_examples.user_query_1)
    target_page = 5
    assert target_page in [page.metadata["source"] for page in retrieved_doc]

    retrieved_doc = my_retriever.invoke(test_examples.user_query_2)
    target_page = 8
    assert target_page in [page.metadata["source"] for page in retrieved_doc]

def test_pg_retrieval_k(my_pg_retriever):
    retrieved_doc = my_pg_retriever.retrieve(query=test_examples.user_query_1,
                                             k_retrieval=top_k_retrieved)
    assert len(retrieved_doc) == top_k_retrieved

def test_pg_retriever_page(my_pg_retriever):
    retrieved_doc = my_pg_retriever.retrieve(query=test_examples.user_query_1,
                                             k_retrieval=top_k_retrieved)
    target_page = 5
    assert target_page in [page.metadata["source"] for page in retrieved_doc]

    retrieved_doc = my_pg_retriever.retrieve(query=test_examples.user_query_2,
                                             k_retrieval=top_k_retrieved)
    target_page = 8
    assert target_page in [page.metadata["source"] for page in retrieved_doc]


