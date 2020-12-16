import os
import json
import warnings
from collections import defaultdict

from dataset import Dataset
from model import Model
from config import config

import tensorflow as tf
import tensorflow.keras.backend as K

import subprocess

class ModelCompare:

	def __init__(self, model1, model2):
		self.model1 = Model(model1)
		self.model2 = Model(model2)
		self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


	def __str__(self):
		return 'Model 1: ' + self.model1.name + '\n' + 'Model 2: ' + self.model2.name + '\n'


	def run_tasks(self):
		if not os.path.exists('outputs'):
			os.makedirs('outputs')
		Model.prepare()
		TASK_MAP = {'sentiment': self.sentiment, 'multilabel': self.multilabel_classification, 'qna': self.qna}
		op_name = ''
		for task in config['tasks']:
			if config['tasks'][task]['do_task']:
				op_name = op_name + task + '_' + str(config['tasks'][task]['epochs']) + '_'
				TASK_MAP[task]()
		with open('outputs/' + op_name[:-1] + '.json', 'w') as fp:
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
		if ft:
			train_dataset = Dataset(dataset, split=Dataset.TRAIN_STR)
		val_dataset = Dataset(dataset, split=Dataset.VALIDATION_STR)

		opt = tf.keras.optimizers.Adam(learning_rate=Model.LEARNING_RATE, epsilon=1e-08, clipnorm=1.0)
		loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

		for i in (self.model1, self.model2):
			model = i.load_model(cls_type, train_dataset.get_num_classes(label_column=label_column))
			model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

			if ft:
				tf_train_data = train_dataset.classification_tokenize(i.tokenizer, Model.BATCH_SIZE, 
																		i.name, text_column=text_column, label_column=label_column)
				model.fit(tf_train_data, epochs=epochs)

				if distil:
					model_teacher = i.load_model(cls_type, train_dataset.get_num_classes(label_column=label_column), for_distillation=True)
					model_teacher.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
					model_teacher.fit(tf_train_data, epochs=epochs)
					model_soft_labels = model_teacher.predict(tf_train_data)
					student_dataset, encoder = train_dataset.student_dataset_encoder(model_soft_labels, text_column=text_column, label_column=label_column)
					model_student = Model.student_model(cls_type, encoder, train_dataset.get_num_classes(label_column=label_column), temperature=Model.TEMPERATURE)

					student_loss_fn = {'soft': Model.get_distillation_loss_fn(), 'hard': Model.get_distillation_loss_fn()}
					loss_wts = {'soft': 1 - Model.ALPHA, 'hard': Model.ALPHA}
					model_student.compile(optimizer=opt, loss=student_loss_fn, loss_weights=loss_wts, metrics=metrics)

					model_student.fit(student_dataset, epochs=epochs)

					del model_soft_labels
					del student_dataset

				del tf_train_data

			tf_val_data = val_dataset.classification_tokenize(i.tokenizer, Model.BATCH_SIZE, 
																i.name, text_column=text_column, label_column=label_column)
			model_eval = {k:v for k, v in zip(model.metrics_names, model.evaluate(tf_val_data, verbose=0))}
			self.results[cls_type][i.name] = model_eval
			self.results[cls_type][i.name]['f1'] = (2 * model_eval['precision'] * model_eval['recall']) / (model_eval['precision'] + model_eval['recall'])
			del self.results[cls_type][i.name]['precision']
			del self.results[cls_type][i.name]['recall']

			if distil:
				model_soft_labels = model.predict(tf_val_data)
				val_student_dataset, encoder = val_dataset.student_dataset_encoder(model_soft_labels, text_column=text_column, label_column=label_column)
				model_eval = {k.split('_')[-1]:v for k, v in zip(model_student.metrics_names, model_student.evaluate(val_student_dataset, verbose=0)) if 'soft' not in k}
				self.results[cls_type]['distilled-' + i.name] = model_eval
				self.results[cls_type]['distilled-' + i.name]['f1'] = (2 * model_eval['precision'] * model_eval['recall']) / (model_eval['precision'] + model_eval['recall'])
				del self.results[cls_type]['distilled-' + i.name]['precision']
				del self.results[cls_type]['distilled-' + i.name]['recall']

				del tf_val_data
				del val_student_dataset

			Model.clean_up()


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
		self.results['qna'] = {model1_name: d1['qna'][model1_name], model2_name: d2['qna'][model2_name]}


if __name__ == '__main__':
	f = ModelCompare(config['model1'], config['model2'])
	f.run_tasks()