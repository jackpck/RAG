from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import init_chat_model
from langchain_core.runnables.base import RunnableSerializable, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class ComponentChainer:
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

    def chain(self, reranked_retriever: VectorStoreRetriever,
              SYSTEM_PROMPT: str) -> RunnableSerializable:
        with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            prompt = f.read()

        rag_prompt = ChatPromptTemplate.from_template(prompt)
        rag_chain = (
            {
                "context": reranked_retriever,
                "question": RunnablePassthrough(),
            }
            | rag_prompt | self.llm
        )
        # TODO: replace context by chunker | embedder | retriever and do away with
        # TODO: chaining manually in main.py
        return rag_chain

