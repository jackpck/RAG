from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

class ChunkRetriever:
    def __init__(self, retriever_search_type: str,
                 retriever_search_kwargs: dict):
        self.retriever_search_type = retriever_search_type
        self.retriever_search_kwargs = retriever_search_kwargs

    def retrieve(self, vectorstore: FAISS) -> VectorStoreRetriever:
        retriever = vectorstore.as_retriever(search_type=self.retriever_search_type,
                                             search_kwargs=self.retriever_search_kwargs)
        return retriever



