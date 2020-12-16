config = {
    'model1': 'roberta',
    'model2': 'bert',
    'tasks': {
        'multilabel': {
            'do_task': True,
            'ft': True,
            'epochs': 20,
            'dataset': 'joelito/sem_eval_2010_task_8',
            'distillation': False
        },
        'sentiment': {
            'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'rotten_tomatoes',
            'distillation': False
        },
        'qna': {
            'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'squad',
        },
    }
}