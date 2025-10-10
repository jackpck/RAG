## Debug

### set PATH for ollama
export PATH="$PATH:/cygdrive/c/Users/jacky/AppData/Local/Programs/Ollama"

before running

ollama run gemma3

### set PATH for model pulled from ollama to D drive

export OLLAMA_MODELS="/cygdrive/d/cygdrive/d/ollama/models"

this is not sufficient in Windows. One need to set the environment variable in the **Edit the 
system environment variables**


### Dynamical chaining using yaml

### get("params", {}) format error

in yaml, if indentation is followed by -, using get("params", {})
will create list of size-one dict instead of dict

### no attribute "__pydantic_fields_set__" error
Remember to set default when setting type setting variables in a class.

### langsmith for experimentation
Less flexible than expected. E.g. not easy to tag each evaluation experiment with the model
metadata. Also prompt experimentation require deploying the RAG as runnable on langsmith,
and this requires a paid subscription :(

### mlflow best practice
- The guiding principle for creating an experiment is the consistency of the input data. 
  If multiple runs use the same input dataset (even if they utilize different portions of it), 
  they logically belong to the same experiment. For other hierarchical categorizations, 
  using tags is advisable.
  
when using subprocess, using "python" does not use the right venv. Use `sys.executable` instead

in order to run `create_dataset()`, first create a SQL db using the following command in the 
root directory:

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

This will create a mlflow.db in the root directory. Then add `mlflow.set_tracking_uri("sqlite:///mlflow.db")
` in the script before running `create_dataset()`


do not do `await llm.ainvoke(query).content` as `await` only bind to `llm.ainvoke` but not `content`.
do `response = await llm.ainvoke(query)` and `response = response.content` instead.

Run `markitdown path/to/pdf > markdown_name.md` to convert pdf to markdown. Preferred
if pdf has many tables and changing layout over pages.

Run below if sh run_docker_image.sh gives line break error
tr -d '\r' < run_docker_image.sh > fixed.sh && mv fixed.sh run_docker_image.sh