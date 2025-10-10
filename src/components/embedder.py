from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import os

from src.utils.sql_connection import NeonPostgres

class DocEmbedder:
    def __init__(self, model_name: str,
                 vs_name: str):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vs_name = vs_name

    def embed(self, split_docs: List[Document],
              persist_vs: str = None, **kwargs) -> FAISS:
        """
        :param split_docs: chunks from the chunker
        :param persist_vs: either 1) local, 2) serverless (postgres database w/ pgvector) or 3) None
        :return:
        """
        vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        if persist_vs == "local":
            vectorstore.save_local(self.vs_name)
        elif persist_vs == "serverless":
            neon_db = NeonPostgres(dbname=os.environ["DB_NAME"].rstrip(),
                                  user=os.environ["DB_USER"].rstrip(),
                                  password=os.environ["DB_PWD"].rstrip(),
                                  host=os.environ["DB_HOST"].rstrip(),
                                  port=os.environ["DB_PORT"].rstrip(),
                                  sslmode="require")
            neon_db.create_embedding_table(tablename=kwargs["tablename"])
            docs_ids, embeddings, doc_metadata, doc_content = neon_db.get_embedding_from_vectorstore(vectorstore)
            neon_db.insert_embedding_to_table(tablename=kwargs["tablename"],
                                              doc_ids=docs_ids,
                                              embeddings=embeddings,
                                              doc_content=doc_content,
                                              doc_metadata=doc_metadata)
            neon_db.commit()
            neon_db.close()
        elif not persist_vs:
            pass

        return vectorstore

    def from_vs(self) -> VectorStoreRetriever:
        """
        only use this for local vectorstore.
        For postgres database with pgvector, retrieve directly from the query
        :return:
        """
        vectorstore = FAISS.load_local(self.vs_name,
                                       self.embedding_model,
                                       allow_dangerous_deserialization=True)

        return vectorstore

    def from_postgres(self, tablename: str) -> str:
        retrieval_query = """
            SELECT document_id, content
            FROM {0}
            ORDER BY embedding <-> %s
            LIMIT {1};
        """.format(tablename, "{0}")
        return retrieval_query

if __name__ == "__main__":
    from src.components.runner import ChainRunner

    CONFIG_PATH = "configs/embedding_pipeline_config.yaml"
    user_query = "dummy"

    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    response = RAG_chain.run(user_query)
    print(f"response:{response}")

