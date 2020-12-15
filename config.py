config = {
    'model1': 'bert',
    'model2': 'roberta',
    'tasks': {
        'multilabel': {
            'do_task': True,
            'ft': True,
            'epochs': 1,
            'dataset': 'joelito/sem_eval_2010_task_8',
            'distillation': True
        },
        'sentiment': {
            'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'rotten_tomatoes',
            'distillation': False
        },
        'qna': {
            'do_task': True,
            'ft': False,
            'epochs': 1,
            'dataset': 'squad',
            'distillation': False
        },
    }
}