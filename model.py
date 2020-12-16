from transformers import logging
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Activation, Embedding, Bidirectional, LSTM, Dense, Lambda

# logging.set_verbosity(logging.CRITICAL)

class Model:


	MODEL_MAP = {'bert': ('bert-base-uncased', ''), 'roberta': ('roberta-base', ''), 'xlnet': ('xlnet-base-cased', '')}

	MODEL_INPUTS = {'bert': ['input_ids', 'token_type_ids', 'attention_mask'], 
					'roberta': ['input_ids', 'attention_mask'], 'xlnet': ['input_ids', 'attention_mask']}

	BATCH_SIZE = 32
	MAX_SEQ_LEN = 128
	LEARNING_RATE = 3e-5

	ALPHA = 0.1
	TEMPERATURE = 2


	def __init__(self, name):
		self.name = name
		self.classification_model = TFAutoModelForSequenceClassification
		self.qna_model = TFAutoModelForQuestionAnswering
		self.type = self.MODEL_MAP[self.name]
		self.tokenizer = AutoTokenizer.from_pretrained(self.type[0])


	@staticmethod
	def student_model(task, encoder, num_classes, temperature=2):
		if task == 'sentiment':
			x = Input(shape=(1,), dtype=tf.string)
			y = encoder(x)
			y = Embedding(
			        input_dim=len(encoder.get_vocabulary()),
			        output_dim=64,
			        mask_zero=True)(y)
			y = Bidirectional(LSTM(64))(y)
			y = Dense(64, activation='relu')(y)
			y = Dense(num_classes)(y)
			y1 = Lambda(lambda x: x / temperature)(y)
			y1 = Activation('softmax', name='soft')(y1)
			y2 = Activation('softmax', name='hard')(y)

			model = tf.keras.Model(x, [y1, y2])
			return model
		if task == 'multilabel':
			x = Input(shape=(1,), dtype=tf.string)
			y = encoder(x)
			y = Embedding(
			        input_dim=len(encoder.get_vocabulary()),
			        output_dim=64,
			        mask_zero=True)(y)
			y = Bidirectional(LSTM(64))(y)
			y = Dense(64, activation='relu')(y)
			y = Dense(num_classes)(y)
			y1 = Lambda(lambda x: x / temperature)(y)
			y1 = Activation('sigmoid', name='soft')(y1)
			y2 = Activation('sigmoid', name='hard')(y)

			model = tf.keras.Model(x, [y1, y2])
			return model


	@staticmethod
	def get_distillation_loss_fn():
		def loss_fn(y_true, y_pred):
			y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
			loss = -1 * (1 - y_pred) * y_true * K.log(y_pred) - y_pred * (1 - y_true) * K.log(1 - y_pred)
			return loss
		return loss_fn

		
	@staticmethod
	def prepare():
		try:
			gpu_list = tf.config.list_physical_devices('GPU')
			for gpu in gpu_list:
				tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.set_soft_device_placement(True)
		except RuntimeError:
			pass


	@staticmethod
	def clean_up():
		K.clear_session()


	def load_model(self, task, num_classes, for_distillation=False):
		if task == 'sentiment':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((self.MAX_SEQ_LEN,), dtype='int32', name=ip))
			base_model = self.classification_model.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			if for_distillation:
				y = Lambda(lambda x: x / self.TEMPERATURE)(y)
			y = Activation('softmax')(y)

			model = tf.keras.Model(inputs, y)
			return model
		elif task == 'multilabel':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((self.MAX_SEQ_LEN,), dtype='int32', name=ip))
			base_model = self.classification_model.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			if for_distillation:
				y = Lambda(lambda x: x / self.TEMPERATURE)(y)
			y = Activation('sigmoid')(y)

			model = tf.keras.Model(inputs, y)
			return model
