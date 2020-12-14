config = {
    'model1': 'bert',
    'model2': 'roberta',
    'tasks': {
        'multilabel': {
        	'do_task': True,
            'ft': True,
            'epochs': 1,
            'dataset': 'joelito/sem_eval_2010_task_8'
        },
        'sentiment': {
        	'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'rotten_tomatoes'
        },
        'multiclass': {
        	'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': ''
        },
        'qna': {
        	'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'squad'
        },
    }
}