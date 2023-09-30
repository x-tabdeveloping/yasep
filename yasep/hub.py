DEFAULT_README = """
---
language:
    - en
tags:
    - embeddings
    - tokenizers
library_name: yasep
---

# {repo}

This repository contains an embedding pipeline that has been trained using [yasep](https://github.com/x-tabdeveloping/yasep)

## Usage
```python
# pip install yasep

from yasep.pipeline import Pipeline

nlp = Pipeline.from_hub('{repo}')
nlp("A text you want to process")

```
"""
