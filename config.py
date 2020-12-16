config = {
    'model1': 'bert',
    'model2': 'xlnet',
    'tasks': {
        'multilabel': {
            'do_task': True,
            'ft': True,
            'epochs': 5,
            'dataset': 'joelito/sem_eval_2010_task_8',
            'distillation': True
        },
        'sentiment': {
            'do_task': False,
            'ft': True,
            'epochs': 5,
            'dataset': 'imdb',
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