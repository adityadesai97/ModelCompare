import os
from datasets import load_dataset, list_datasets, logging

import tensorflow as tf

from model import Model

import numpy as np

# logging.set_verbosity(logging.CRITICAL)

class Dataset:

	HF_DATASETS = list_datasets()
	DATA_PATH = '../data/'

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
			if self.split == 'validation':
				try:
					return load_dataset(self.name, split='validation')
				except ValueError:
					pass
				try:
					return load_dataset(self.name, split='test')
				except ValueError:
					raise RuntimeError('Invalid dataset. No validation set found.')
			else:
				return load_dataset(self.name, split=self.split)
		else:
			filename = os.path.join(self.DATA_PATH, self.name, str(self.split) + '.' + str(self.type))
			return load_dataset(self.type, data_files=filename)


	def classification_tokenize(self, tokenizer, batch_size, model_name, text_label='text', label_column='label'):
		def encode(example):
			return tokenizer(example[text_label], padding='max_length', truncation=True)
		dataset = self.data.map(encode)
		dataset.set_format(type='tensorflow', columns=Model.MODEL_INPUTS[model_name]+[label_column])
		features = {x: dataset[x].to_tensor(default_value=0, shape=(None, Model.MAX_SEQ_LEN)) for x in Model.MODEL_INPUTS[model_name]}
		labels = tf.keras.utils.to_categorical(dataset[label_column], num_classes=self.get_num_classes(label_column=label_column))
		tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
		return tfdataset
