from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List

from src.utils.sql_connection import NeonPostgres

class ChunkRetriever:
    def __init__(self, retriever_search_type: str,
                 retriever_search_kwargs: dict):
        self.retriever_search_type = retriever_search_type
        self.retriever_search_kwargs = retriever_search_kwargs

    def retrieve(self, vectorstore: FAISS) -> VectorStoreRetriever:
        retriever = vectorstore.as_retriever(search_type=self.retriever_search_type,
                                             search_kwargs=self.retriever_search_kwargs)
        return retriever

    def retrieve_postgres(self, retrieval_query: str,
                          k: int) -> str:
        retrieval_query_k = retrieval_query.format(k)
        return retrieval_query_k


class PostgresRetriever:
    def __init__(self, tablename: str,
                 embedding_model_name: str):
        self.neon_db = NeonPostgres(dbname=os.environ["DB_NAME"].rstrip(),
                                    user=os.environ["DB_USER"].rstrip(),
                                    password=os.environ["DB_PWD"].rstrip(),
                                    host=os.environ["DB_HOST"].rstrip(),
                                    port=os.environ["DB_PORT"].rstrip(),
                                    sslmode="require")
        self.tablename = tablename
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def retrieve(self, query: str, k_retrieval: int) -> List[Document]:
        retrieval_query = f"""
            SELECT document_id, content, metadata AS page
            FROM {self.tablename}
            ORDER BY embedding <-> %s::vector
            LIMIT {k_retrieval};
        """
        query_embedding = self.embedding_model.embed_query(query)
        self.neon_db.cur.execute(retrieval_query, (query_embedding,))
        retrieved_docs = self.neon_db.cur.fetchall()
        return [Document(page_content=doc[1], metadata={"source":int(doc[2])}) for doc in retrieved_docs]


if __name__ == "__main__":
    tablename = "faiss_index_google_genai_risk_mgmt"
    model_name = "all-MiniLM-L6-v2"
    retriever = PostgresRetriever(tablename=tablename,
                                  embedding_model_name=model_name)

    user_input = "What are the top risks in GenAI?"
    top_k = 5
    df_retrieved = retriever.retrieve(query=user_input,
                                       k_retrieval=top_k)

    for row in df_retrieved:
        print(row)
        print("*"*40)

