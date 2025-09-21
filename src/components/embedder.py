from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List
from langchain_core.documents import Document

class DocEmbedder:
    def __init__(self, model_name: str,
                 vs_name: str):
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name)
        self.vs_name = vs_name

    def embed(self, split_docs: List[Document],
              persist_vs: bool) -> FAISS:
        vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        if persist_vs:
            vectorstore.save_local(self.vs_name)
        return vectorstore

    def from_vs(self) -> FAISS:
        vectorstore = FAISS.load_local(self.vs_name,
                                       self.embedding_model,
                                       allow_dangerous_deserialization=True)
        return vectorstore
