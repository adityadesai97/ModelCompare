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
		distil = config['tasks']['sentiment']['distillation']
		self.classification('sentiment', ft, dataset, epochs, distil, 'text', 'label')


	def multilabel_classification(self):
		ft = config['tasks']['multilabel']['ft']
		dataset = config['tasks']['multilabel']['dataset']
		epochs = config['tasks']['multilabel']['epochs']
		distil = config['tasks']['multilabel']['distillation']
		self.classification('multilabel', ft, dataset, epochs, distil, 'sentence', 'relation')


	def classification(self, cls_type, ft, dataset, epochs, distil, text_column='text', label_column='label'):
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
		metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

		model1.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
		model2.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

		if ft:
			tf_train_data_1 = train_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
																	self.model1.name, text_column=text_column, label_column=label_column)
			model1.fit(tf_train_data_1, epochs=epochs)

			if distil:
				model1_soft_labels = model1.predict(tf_train_data_1)
				student_dataset, encoder = train_dataset.student_dataset_encoder(model1_soft_labels, text_column=text_column, label_column=label_column)
				# distillation_dataset = tf.data.Dataset.zip((tf_train_data_1, student_dataset)).batch(Model.BATCH_SIZE)
				# print(next(iter(student_dataset)))
				model1_student = Model.student_model(cls_type, encoder, train_dataset.get_num_classes(label_column=label_column))

				loss_fn = {'soft': Model.get_distillation_loss_fn(), 'hard': Model.get_distillation_loss_fn()}
				loss_wts = {'soft': Model.ALPHA, 'hard': 1 - Model.ALPHA}
				model1_student.compile(optimizer=opt, loss=loss_fn, loss_weights=loss_wts, metrics=metrics)

				# print(next(iter(student_dataset)))

				model1_student.fit(student_dataset, epochs=epochs)

			del tf_train_data_1

			tf_train_data_2 = train_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
																	self.model2.name, text_column=text_column, label_column=label_column)
			model2.fit(tf_train_data_2, epochs=epochs)

			# if distil:
			# 	model2_soft_labels = model2.predict(tf_train_data_2)
			# 	model2_student = Model.student_model()

			# 	loss_fn = Model.distillation_loss
			# 	model1_student.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

			del tf_train_data_2

		tf_val_data_1 = val_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
															self.model1.name, text_column=text_column, label_column=label_column)
		tf_val_data_2 = val_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
															self.model2.name, text_column=text_column, label_column=label_column)
		model1_eval = {k:v for k, v in zip(model1.metrics_names, model1.evaluate(tf_val_data_1, verbose=0))}
		model2_eval = {k:v for k, v in zip(model2.metrics_names, model2.evaluate(tf_val_data_2, verbose=0))}
		self.results[cls_type][self.model1.name] = model1_eval
		self.results[cls_type][self.model2.name] = model2_eval

		# if config['tasks'][cls_type]['distillation']:
			# model1_student = Model.load_student_model()
			# model2_student = Model.load_student_model()

			# model1.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
			# model2.compile(optimizer=opt, loss=loss_fn, metrics=metrics)





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