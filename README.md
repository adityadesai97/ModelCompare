# ModelCompare
### Class project for Practical Deep Learning System Performance.
A framework to compare 2 large language models.

Models supported:
1. BERT
2. XLNet
3. RoBERTa.

Tasks supported:
1. Multilabel Classification
2. Sentiment Classification
3. Question Answering

Datasets supported:
1. Multilabel Classification - sem_eval_2010_task_8
2. Sentiment Classification - rotten_tomatoes
3. Question Answering: SQuAD

Training hyperparameters like batch size and learning rate can be changed in models.py (BATCH_SIZE, LEARNING_RATE)
Classification tasks also support a distillation feature. Distillation hyperparameters can be changed in models.py (ALPHA, TEMPERATURE).

# Installation:
```
git clone https://github.com/adityadesai97/ModelCompare.git
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
