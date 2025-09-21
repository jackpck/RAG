from typing import List
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.vectorstores.base import VectorStoreRetriever

class Reranker:
    k_rerank: int = 3
    model_rerank: str = None
    model_rerank_provider: str = None
    temperature_rerank: float = 0
    top_k_rerank: int = 1
    top_p_rerank: float = 0.9
    reranker_prompt_path: str = None

    def __init__(self,
                 k_rerank: int,
                 model_rerank: str,
                 model_rerank_provider: str,
                 temperature_rerank: float,
                 top_k_rerank: int,
                 top_p_rerank: float,
                 reranker_prompt_path: str):
        self.k_rerank = k_rerank
        self.model_rerank = model_rerank
        self.model_rerank_provider = model_rerank_provider
        self.temperature_rerank = temperature_rerank
        self.top_k_rerank = top_k_rerank
        self.top_p_rerank = top_p_rerank
        self._setup_prompt(reranker_prompt_path=reranker_prompt_path)

    def _setup_prompt(self, reranker_prompt_path):
        with open(reranker_prompt_path, "r", encoding="utf-8") as f:
            self.reranker_prompt = f.read()

    def rerank(self, retriever: VectorStoreRetriever,
               query: str) -> List[Document]:
        retrieved_docs = retriever.invoke(query)
        rerank_llm = init_chat_model(model=self.model_rerank,
                                   model_provider=self.model_rerank_provider,
                                   temperature=self.temperature_rerank,
                                   top_k=self.top_k_rerank,
                                   top_p=self.top_p_rerank)

        ranked = []
        for doc in retrieved_docs:
            try:
                response = rerank_llm.invoke(self.reranker_prompt.format(query,
                                                                    doc.page_content)).strip()
                score = int(response)
            except:
                score = 0
            ranked.append((score, doc))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.k_rerank]]


