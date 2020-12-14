import json
import warnings
from collections import defaultdict

from dataset import Dataset
from model import Model
from config import config

import tensorflow as tf

class ModelCompare:

	def __init__(self, model1, model2):
		self.model1 = Model(model1)
		self.model2 = Model(model2)
		self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


	def __str__(self):
		return 'Model 1: ' + self.model1.name + '\n' + 'Model 2: ' + self.model2.name + '\n'


	def run_tasks(self):
		TASK_MAP = {'sentiment': self.sentiment, 'multiclass': self.multiclass_classification, 'qna': self.qna}
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

		opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
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
		ft = config['tasks']['qna']['ft']
		dataset = config['tasks']['qna']['dataset']
		epochs = config['tasks']['qna']['epochs']

		if dataset != 'squad':
			warning.warn('Only SQuAD is currently supported for QnA. Defaulting to SQuAD')
		if ft:
			train_dataset = Dataset('squad', split=self.TRAIN_STR)
		val_dataset = Dataset('squad', split=self.VALIDATION_STR)

		model1 = self.model1.load_model(self.model1.qna_model, 'qna', train_dataset.get_num_classes(label_column='relation'))


if __name__ == '__main__':
	f = ModelCompare(config['model1'], config['model2'])
	f.run_tasks()