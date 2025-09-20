from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from typing import List

class Reranker:
    def __init__(self, model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float):
        self.llm = init_chat_model(model=model,
                                   model_provider=model_provider,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)
        self.prompt = f"""
            Rate the relevance of the following document to the query on a scale
            of 1 (irrelevant) to 10 (highly relevant).

            Query: '{1}'
            
            Document:
            \"\"\"
            {2}
            \"\"\"
            
            Respond with only the number.
            """
    def rerank(self, query: str,
               documents: List[Document],
               top_k: int) -> List[Document]:
        ranked = []
        for doc in documents:
            try:
                response = llm.invoke(self.prompt.format(query,
                                                         doc.page_content)).strip()
                score = int(response)
            except:
                score = 0
            ranked.append((score, doc))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
