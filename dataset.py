import os
from datasets import load_dataset, list_datasets, logging

import tensorflow as tf

from model import Model

import numpy as np

# logging.set_verbosity(logging.CRITICAL)

class Dataset:

	HF_DATASETS = list_datasets()
	DATA_PATH = '../data/'

	TRAIN_STR = 'train'
	TEST_STR = 'test'
	VALIDATION_STR = 'validation'

	def __init__(self, name, split):
		self.name = name
		self.split = split
		if self.name not in self.HF_DATASETS:
			self.type = 'csv'
		else:
			self.type = 'hf'

		self.data = self.get_dataset()


	def get_num_classes(self, label_column='label'):
		return self.data.features[label_column].num_classes


	def get_dataset(self):
		if self.type == 'hf':
			if self.split == self.VALIDATION_STR:
				try:
					return load_dataset(self.name, split=self.VALIDATION_STR)
				except ValueError:
					pass
				try:
					return load_dataset(self.name, split=self.TEST_STR)
				except ValueError:
					raise RuntimeError('Invalid dataset. No validation set found.')
			else:
				return load_dataset(self.name, split=self.split)
		else:
			filename = os.path.join(self.DATA_PATH, self.name, str(self.split) + '.' + str(self.type))
			return load_dataset(self.type, data_files=filename)


	def student_dataset_encoder(self, soft_labels, text_column='text', label_column='label'):
		dataset = self.data
		dataset.set_format(type='tensorflow', columns=[text_column])
		features = dataset[text_column]
		# .to_tensor(default_value=0, shape=(None, Model.MAX_SEQ_LEN))
		hard_labels = tf.keras.utils.to_categorical(dataset[label_column], num_classes=self.get_num_classes(label_column=label_column))
		labels = {'soft': soft_labels, 'hard': hard_labels}
		tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(Model.BATCH_SIZE)
		VOCAB_SIZE = 30522
		encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
		    max_tokens=VOCAB_SIZE)
		encoder.adapt(tfdataset.map(lambda text, label: text))

		return tfdataset, encoder


	def classification_tokenize(self, tokenizer, batch_size, model_name, text_column='text', label_column='label'):
		def encode(example):
			return tokenizer(example[text_column], padding='max_length', truncation=True)
		dataset = self.data.map(encode)
		dataset.set_format(type='tensorflow', columns=Model.MODEL_INPUTS[model_name]+[label_column])
		features = {x: dataset[x].to_tensor(default_value=0, shape=(None, Model.MAX_SEQ_LEN)) for x in Model.MODEL_INPUTS[model_name]}
		labels = tf.keras.utils.to_categorical(dataset[label_column], num_classes=self.get_num_classes(label_column=label_column))
		tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(Model.BATCH_SIZE)
		return tfdataset
