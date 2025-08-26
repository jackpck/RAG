from langchain.chat_models import ChatOllama

class CheckAnswer:
    def __init__(self, model_checker: str,
                 temperature_checker: float,
                 top_k_checker: int,
                 top_p_checker: float):
        self.llm = ChatOllama(model=model_checker,
                              temperature=temperature_checker,
                              top_k=top_k_checker,
                              top_p=top_p_checker)
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

    def check_response_answer(self, QUERY_ANSWER: str,
                              rag_response: str) -> int:
        with open(QUERY_ANSWER, "r", encoding="utf-8") as f:
            answer_prompt = f.read()

        try:
            print('run llm')
            response = llm.invoke(self.prompt.format(answer_prompt,
                                                     rag_response)).strip()
            score = int(response)
        except:
            print('except')
            score = 0

        return score


