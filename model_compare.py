import json
import warnings
from collections import defaultdict

from dataset import Dataset
from model import Model
from config import config

import tensorflow as tf

import subprocess

class ModelCompare:

	def __init__(self, model1, model2):
		self.model1 = Model(model1)
		self.model2 = Model(model2)
		self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


	def __str__(self):
		return 'Model 1: ' + self.model1.name + '\n' + 'Model 2: ' + self.model2.name + '\n'


	def run_tasks(self):
		TASK_MAP = {'sentiment': self.sentiment, 'multilabel': self.multilabel_classification, 'qna': self.qna}
		for task in config['tasks']:
			if config['tasks'][task]['do_task']:
				TASK_MAP[task]()
		with open('results.json', 'w') as fp:
		    json.dump(self.results, fp)


	def sentiment(self):
		ft = config['tasks']['sentiment']['ft']
		dataset = config['tasks']['sentiment']['dataset']
		epochs = config['tasks']['sentiment']['epochs']
		self.classification('sentiment', ft, dataset, epochs, 'text', 'label')


	def multilabel_classification(self):
		ft = config['tasks']['multilabel']['ft']
		dataset = config['tasks']['multilabel']['dataset']
		epochs = config['tasks']['multilabel']['epochs']
		self.classification('multilabel', ft, dataset, epochs, 'sentence', 'relation')


	def classification(self, cls_type, ft, dataset, epochs, text_column, label_column):
		if isinstance(dataset, tuple):
			if ft:
				train_dataset = Dataset(dataset[0], dataset[1], split=Dataset.TRAIN_STR)
			val_dataset = Dataset(dataset[0], dataset[1], split=Dataset.VALIDATION_STR)
		else:
			if ft:
				train_dataset = Dataset(dataset, split=Dataset.TRAIN_STR)
			val_dataset = Dataset(dataset, split=Dataset.VALIDATION_STR)
		
		model1 = self.model1.load_model(self.model1.classification_model, cls_type, train_dataset.get_num_classes(label_column=label_column))
		model2 = self.model2.load_model(self.model2.classification_model, cls_type, train_dataset.get_num_classes(label_column=label_column))

		opt = tf.keras.optimizers.Adam(learning_rate=Model.LEARNING_RATE, epsilon=1e-08, clipnorm=1.0)
		loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		metrics = ['accuracy', 'f1']

		model1.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
		model2.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

		if ft:
			tf_train_data_1 = train_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
																	self.model1.name, text_column=text_column, label_column=label_column)
			model1.fit(tf_train_data_1, epochs=epochs)

			del tf_train_data_1

			tf_train_data_2 = train_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
																	self.model2.name, text_column=text_column, label_column=label_column)
			model2.fit(tf_train_data_2, epochs=epochs)

		tf_val_data_1 = val_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
															self.model1.name, text_column=text_column, label_column=label_column)
		tf_val_data_2 = val_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
															self.model2.name, text_column=text_column, label_column=label_column)
		model1_eval = {k:v for k, v in zip(model1.metric_names, model1.evaluate(tf_val_data_1, verbose=0))}
		model2_eval = {k:v for k, v in zip(model2.metric_names, model2.evaluate(tf_val_data_2, verbose=0))}
		self.results[cls_type][self.model1.name] = model1_eval
		self.results[cls_type][self.model2.name] = model2_eval


	def qna(self): 
#         config_qa = {"--model_name_or_path": Model.MODEL_MAP[self.model1.name], \
#                      "--dataset_name": dataset, \
#                      "--do_train": ft, \
#                      "--do_eval": True, \
#                      "--per_device_train_batch_size": Model.BATCH_SIZE, \
#                      "--learning_rate": Model.LEARNING_RATE, \
#                      "--num_train_epochs": epochs, \
#                      "--max_seq_length": Model.MAX_SEQ_LEN, \
#                      "--doc_stride": 128, \
#                      "--output_dir": "/tmp/debug_squa" \
#                     }
            
# 		with open('config_qa.json', 'w', encoding='utf-8') as f:
# 			json.dump(config_qa, f, ensure_ascii=False, indent=4)
		ft = config['tasks']['qna']['ft']
		dataset = config['tasks']['qna']['dataset']
		epochs = config['tasks']['qna']['epochs']
		if dataset != 'squad':
			warning.warn('Only SQuAD is currently supported for QnA. Defaulting to SQuAD')
			dataset = 'squad'
		if ft:
			command1 = ['python', 'run_qa.py', \
						'--model_name_or_path', Model.MODEL_MAP[self.model1.name][0], \
						'--dataset_name', dataset, \
						'--do_train', \
						'--do_eval', \
						'--per_device_train_batch_size', str(Model.BATCH_SIZE), \
						'--learning_rate', str(Model.LEARNING_RATE), \
						'--num_train_epochs', str(epochs), \
						'--max_seq_length', str(Model.MAX_SEQ_LEN), \
						'--doc_stride', '32', \
						'--output_dir', '/home/jupyter/ModelCompare/qna_output']
			command2 = ['python', 'run_qa.py', \
						'--model_name_or_path', Model.MODEL_MAP[self.model2.name][0], \
						'--dataset_name', dataset, \
						'--do_train', \
						'--do_eval', \
						'--per_device_train_batch_size', str(Model.BATCH_SIZE), \
						'--learning_rate', str(Model.LEARNING_RATE), \
						'--num_train_epochs', str(epochs), \
						'--max_seq_length', str(Model.MAX_SEQ_LEN), \
						'--doc_stride', '32', \
						'--output_dir', '/home/jupyter/ModelCompare/qna_output']
		else:
			command1 = ['python', 'run_qa.py', \
						'--model_name_or_path', Model.MODEL_MAP[self.model1.name][0], \
						'--dataset_name', dataset, \
						'--do_eval', \
						'--per_device_train_batch_size', str(Model.BATCH_SIZE), \
						'--learning_rate', str(Model.LEARNING_RATE), \
						'--num_train_epochs', str(epochs), \
						'--max_seq_length', str(Model.MAX_SEQ_LEN), \
						'--doc_stride', '32', \
						'--output_dir', '/home/jupyter/ModelCompare/qna_output']  
			command2 = ['python', 'run_qa.py', \
						'--model_name_or_path', Model.MODEL_MAP[self.model2.name][0], \
						'--dataset_name', dataset, \
						'--do_eval', \
						'--per_device_train_batch_size', str(Model.BATCH_SIZE), \
						'--learning_rate', str(Model.LEARNING_RATE), \
						'--num_train_epochs', str(epochs), \
						'--max_seq_length', str(Model.MAX_SEQ_LEN), \
						'--doc_stride', '32', \
						'--output_dir', '/home/jupyter/ModelCompare/qna_output']            
		p1 = subprocess.run(command1)
		p2 = subprocess.run(command2)
		model1_name = Model.MODEL_MAP[self.model1.name][0].split('-')[0]
		model2_name = Model.MODEL_MAP[self.model2.name][0].split('-')[0]        
		with open('qna_results_' + model1_name + '.json') as f:
			d1 = json.load(f)
		with open('qna_results_' + model2_name + '.json') as f:
			d2 = json.load(f)
		self.results = {'qna': {model1_name: d1['qna'][model1_name], model2_name: d2['qna'][model2_name]}}
        
if __name__ == '__main__':
	f = ModelCompare(config['model1'], config['model2'])
	f.run_tasks()