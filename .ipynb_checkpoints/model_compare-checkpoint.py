from absl import app
from absl import flags
import warnings

from dataset import Dataset
from model import Model

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('model1', 'bert', 'First model')
flags.DEFINE_string('model2', 'roberta', 'Second model')
flags.DEFINE_boolean('sentiment', False, 'Get sentiment analysis results')
flags.DEFINE_boolean('multilabel', False, 'Get multilabel classification results')
flags.DEFINE_boolean('qna', False, 'Get question-answering results')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')

class ModelCompare:
	TRAIN_STR = 'train'
	VALIDATION_STR = 'validation'


	def __init__(self, model1, model2):
		self.model1 = Model(model1)
		self.model2 = Model(model2)


	def __str__(self):
		return 'Model 1: ' + self.model1.name + '\n' + 'Model 2: ' + self.model2.name + '\n'


	def run_tasks(self):
		if FLAGS.sentiment:
			self.sentiment(ft=True)
		if FLAGS.multilabel:
			self.multilabel_classification(ft=True)
		if FLAGS.qna:
			self.qna(ft=True)


	def sentiment(self, dataset='rotten_tomatoes', ft=False):
		if ft:
			train_dataset = Dataset(dataset, split=self.TRAIN_STR)
		val_dataset = Dataset(dataset, split=self.VALIDATION_STR)
		
		model1 = self.model1.load_model(self.model1.classification_model, 'sentiment', train_dataset.get_num_classes())
		model2 = self.model2.load_model(self.model2.classification_model, 'sentiment', train_dataset.get_num_classes())

		opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
		loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

		model1.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
		model2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

		if ft:
			tf_train_data_1 = train_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
																	self.model1.name)
			model1.fit(tf_train_data_1, epochs=FLAGS.epochs)

			del tf_train_data_1

			tf_train_data_2 = train_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
																	self.model2.name)
			model2.fit(tf_train_data_2, epochs=FLAGS.epochs)

		tf_val_data_1 = val_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
															self.model1.name)
		tf_val_data_2 = val_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
															self.model2.name)
		print(self.model1.name + ' accuracy = ' + str(model1.evaluate(tf_val_data_1, verbose=0)[1]))
		print(self.model2.name + ' accuracy = ' + str(model2.evaluate(tf_val_data_2, verbose=0)[1]))


	def multilabel_classification(self, dataset='joelito/sem_eval_2010_task_8', ft=False):
		if ft:
			train_dataset = Dataset(dataset, split=self.TRAIN_STR)
		val_dataset = Dataset(dataset, split=self.VALIDATION_STR)
		
		model1 = self.model1.load_model(self.model1.classification_model, 'multilabel', train_dataset.get_num_classes(label_column='relation'))
		model2 = self.model2.load_model(self.model2.classification_model, 'multilabel', train_dataset.get_num_classes(label_column='relation'))

		opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
		loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
		metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

		model1.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
		model2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

		if ft:
			tf_train_data_1 = train_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
																	self.model1.name, text_label='sentence', label_column='relation')
			model1.fit(tf_train_data_1, epochs=FLAGS.epochs)

			del tf_train_data_1

			tf_train_data_2 = train_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
																	self.model2.name, text_label='sentence', label_column='relation')
			model2.fit(tf_train_data_2, epochs=FLAGS.epochs)

		tf_val_data_1 = val_dataset.classification_tokenize(self.model1.tokenizer, Model.BATCH_SIZE, 
															self.model1.name, text_label='sentence', label_column='relation')
		tf_val_data_2 = val_dataset.classification_tokenize(self.model2.tokenizer, Model.BATCH_SIZE, 
															self.model2.name, text_label='sentence', label_column='relation')
		print(self.model1.name + ' accuracy = ' + str(model1.evaluate(tf_val_data_1, verbose=0)[1]))
		print(self.model2.name + ' accuracy = ' + str(model2.evaluate(tf_val_data_2, verbose=0)[1]))


	def qna(self, dataset='squad', ft=False):
		if dataset != 'squad':
			warning.warn('Only SQuAD is currently supported for QnA. Defaulting to SQuAD')
		if ft:
			train_dataset = Dataset('squad', split=self.TRAIN_STR)
		val_dataset = Dataset('squad', split=self.VALIDATION_STR)

		model1 = self.model1.load_model(self.model1.qna_model, 'qna', train_dataset.get_num_classes(label_column='relation'))


	# def multiclass_classification(self, dataset='', ft=False):
	# 	if ft:
	# 		train_data = Dataset(dataset, split=self.TRAIN_STR)
	# 	val_data = Dataset(dataset, split=self.VALIDATION_STR)
	# 	print('c1', dataset, ft)

def main(argv):
	f = ModelCompare(FLAGS.model1, FLAGS.model2)
	f.run_tasks()

if __name__ == '__main__':
	app.run(main)