from typing import List
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.vectorstores.base import VectorStoreRetriever
import asyncio

from src.utils.langsmith_loader import load_prompt
from src.utils.syncify import sync

class Reranker:
    k_rerank: int = 3
    model_rerank: str = None
    model_rerank_provider: str = None
    temperature_rerank: float = 0
    top_k_rerank: int = 1
    top_p_rerank: float = 0.9
    prompt_name: str
    prompt_version: str

    def __init__(self,
                 k_rerank: int,
                 model_rerank: str,
                 model_rerank_provider: str,
                 temperature_rerank: float,
                 top_k_rerank: int,
                 top_p_rerank: float,
                 prompt_name: str,
                 prompt_version: str):
        self.k_rerank = k_rerank
        self.rerank_llm = init_chat_model(model=model_rerank,
                                          model_provider=model_rerank_provider,
                                          temperature=temperature_rerank,
                                          top_k=top_k_rerank,
                                          top_p=top_p_rerank)
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = load_prompt(prompt_name, prompt_version)
        self.reranker_prompt = prompt.format_messages()[0].content

    @sync
    async def rerank(self, retriever: VectorStoreRetriever,
                     query: str) -> List[str]:
        retrieved_docs = retriever.invoke(query)

        async def score_doc(doc):
            try:
                response = await self.rerank_llm.ainvoke(self.reranker_prompt.format(query, doc))
                score = int(response.content)
            except:
                score = 0
            return (score, doc)

        ranked_tasks = [score_doc(doc.page_content) for doc in retrieved_docs]
        ranked = await asyncio.gather(*ranked_tasks)
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.k_rerank]]

if __name__ == "__main__":
    from src.components.retriever import ChunkRetriever
    from src.components.embedder import DocEmbedder
    import os

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

    retriever_param = {
        "retriever_search_type": "similarity",
        "retriever_search_kwargs": {"k": 20}
    }
    embedder_param = {
        "model_name": "all-MiniLM-L6-v2",
        "vs_name": "faiss_index_google_genai_risk_mgmt"
    }
    reranker_param = {
        "k_rerank": 5,  # choose top k reranker score
        "model_rerank": "gemini-2.5-flash",
        "model_rerank_provider": "google_genai",
        "temperature_rerank": 0,
        "top_k_rerank": 5,  # top k token in generation of the rerank score
        "top_p_rerank": 0.8,
        "prompt_name": "system-reranker-prompt",
        "prompt_version": "latest"
    }
    test_query = "What are the Socio-technical Considerations?"

    embedder = DocEmbedder(**embedder_param)
    vectorstore = embedder.from_vs()

    retriever = ChunkRetriever(**retriever_param)
    top_k_retriever = retriever.retrieve(vectorstore=vectorstore)

    reranker = Reranker(**reranker_param)
    context = reranker.rerank(retriever=top_k_retriever,
                              query=test_query)
    print(context)
