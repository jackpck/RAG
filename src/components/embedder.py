from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

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
        :param persist_vs: either 1) local, 2) serverless or 3) None
        :return:
        """
        vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        if persist_vs == "local":
            vectorstore.save_local(self.vs_name)
        elif persist_vs == "serverless":
            neon_db = NeonPostgres(dbname=kwargs["dbname"],
                                  user=kwargs["user"],
                                  password=kwargs["password"],
                                  host=kwargs["host"],
                                  port=kwargs["port"])
            neon_db.create_embedding_table(tablename=kwargs["tablename"])
            docs_ids, embeddings = neon_db.get_embedding_from_vectorstore(vectorstore)
            neon_db.insert_embedding_to_table(tablename=kwargs["tablename"],
                                             doc_ids=docs_ids,
                                             embeddings=embeddings)
            neon_db.commit()
            neon_db.close()
        elif not persist_vs:
            pass

        return vectorstore

    def from_vs(self) -> FAISS:
        # TODO read from serverless
        vectorstore = FAISS.load_local(self.vs_name,
                                       self.embedding_model,
                                       allow_dangerous_deserialization=True)
        return vectorstore
