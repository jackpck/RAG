from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

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

    def retriever(self, query: str, k: int):
        retrieval_query = f"""
            SELECT document_id, content
            FROM {self.tablename}
            ORDER BY embedding <-> %s
            LIMIT {k};
        """
        query_embedding = self.embedding_model.embed_query(query)
        retrieved_doc = self.neon_db.cur.execute(retrieval_query, (query_embedding,))
        return retrieved_doc


if __name__ == "__main__":
    tablename = ""
    model_name = "all-MiniLM-L6-v2"
    retriever = PostgresRetriever(tablename=tablename,
                                  embedding_model_name=model_name)

    user_input = ""
    top_k = 5
    df_retrieved = retriever.retriever(query=user_input,
                                       k=top_k)


