from langchain.chat_models import init_chat_model

class LLMJudge:
    def __init__(self, model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float):
        self.llm_judge = init_chat_model(model=model,
                                   model_provider=model_provider,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p)
        self.prompt = f"""
        Determine if the following response contains the true answer with 0 (not 
        contain) and 1 (contain).

        True answer:
        \"\"\"
        {1}
        \"\"\"

        Response:
        \"\"\"
        {2}
        \"\"\"

        Respond with only the number.
        """

    def accuracy_metric(self, gold_answer: str,
                        rag_response: str) -> bool:
        try:
            response = self.llm_judge.invoke(self.prompt.format(gold_answer,
                                                                rag_response)).strip()
            score = int(response)
        except:
            score = 0

        return bool(score)


