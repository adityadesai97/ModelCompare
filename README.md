# ModelCompare
### Class project for Practical Deep Learning System Performance.
A framework to compare 2 large language models. The language models must be available on Huggingface's Transformers library.

# Installation:
```
git clone 
```

# Usage:
## Set up config.py as follows:
```
config = {
    'model1': '',
    'model2': '',
    'tasks': {
        'multilabel': {
            'do_task': True/False,
            'ft': True,
            'epochs': 1,
            'dataset': 'joelito/sem_eval_2010_task_8',
            'distillation': True/False
        },
        'sentiment': {
            'do_task': True/False,
            'ft': True,
            'epochs': 1,
            'dataset': 'imdb',
            'distillation': True/False
        },
        'qna': {
            'do_task': True/False,
            'ft': True,
            'epochs': 1,
            'dataset': 'squad',
        },
    }
}
```
## Run the code as follows:
```
python model_compare.py
```
