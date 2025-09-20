from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List
from langchain_core.documents import Document

class DocEmbedder:
    def __init__(self, model_name: str):
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=model_name)

    def embed(self, split_docs: List[Document], vs_name: str) -> FAISS:
        vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        if vs_name:
            vectorstore.save_local(vs_name)
        return vectorstore

    def from_vs(self, vs_name: str) -> FAISS:
        vectorstore = FAISS.load_local(vs_name,
                                       self.embedding_model,
                                       allow_dangerous_deserialization=True)
        return vectorstore
