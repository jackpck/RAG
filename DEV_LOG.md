## Debug

### set PATH for ollama
export PATH="$PATH:/cygdrive/c/Users/jacky/AppData/Local/Programs/Ollama"

before running

ollama run gemma3

### set PATH for model pulled from ollama to D drive

export OLLAMA_MODELS="/cygdrive/d/cygdrive/d/ollama/models"

this is not sufficient in Windows. One need to set the environment variable in the **Edit the 
system environment variables**


### Dynamica chaining using yaml

### get("params", {}) format error

in yaml, if indentation is followed by -, using get("params", {})
will create list of size-one dict instead of dict

