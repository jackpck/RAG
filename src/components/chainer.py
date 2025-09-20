from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
              SYSTEM_PROMPT: str) -> RetrievalQA:
        with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            prompt = f.read()

        rag_prompt = {"prompt": PromptTemplate(input_variabls=["context", "question"],
                                               template=prompt)}
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                               retriever=reranked_retriever,
                                               chain_type_kwargs=rag_prompt,
                                               chain_type='stuff',
                                               return_source_documents=True)
        return qa_chain

