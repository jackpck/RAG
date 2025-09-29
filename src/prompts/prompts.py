SYSTEM_PROMPT = """
You are an AI assistant. Answer the question using only the provided context. If
no context is given or if context is empty, Answer with 
'Unable to answer. Question is unanswerable given the scope of the document'.

Context:
{context}

Question: {question}
Answer:
"""

RERANKER_PROMPT = """
Rate the relevance of the following document to the query on a scale
of 1 (irrelevant) to 10 (highly relevant).

Query: '{0}'

Document:
\"\"\"
{1}
\"\"\"

Respond with only the number
"""

AUTORATER_PROMPT = """
We define context as 'sufficient' if it contains all OR partially contains the necessary information 
to provide a definitive answer to the query and 'insufficient' if it lacks the 
necessary information, is inconclusive, or contains contradictory information. 

TASK: Given the context-query pair below, determine if it is sufficient (score = 1) or 
insufficient (score = 0)

Context:
\"\"\"
{0}
\"\"\"

Query:
\"\"\"
{1}
\"\"\"

Response with only the score number.
"""

LLMJUDGE_PROMPT = """
Determine if the following response contains the true answer with 0 (not 
contain) and 1 (contain). If the true answer consists of multiple points, only score 1 if the response
contains ALL the points mentioned in the true answer.

True answer:
\"\"\"
{gold_response}
\"\"\"

Response:
\"\"\"
{llm_response}
\"\"\"

Respond with only the number.
"""
