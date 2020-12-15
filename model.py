from transformers import logging
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering

import tensorflow as tf
from tensorflow.keras.layers import Input, Activation

# logging.set_verbosity(logging.CRITICAL)

class Model:


	MODEL_MAP = {'bert': ('bert-base-uncased', ''), 'distilbert': ('distilbert-base-uncased', ''), 'roberta': ('roberta-base', ''), 'xlnet': ('xlnet-base-uncased', '')}

	MODEL_INPUTS = {'bert': ['input_ids', 'token_type_ids', 'attention_mask'], 'distilbert': ['input_ids', 'token_type_ids', 'attention_mask'], 
					'roberta': ['input_ids', 'attention_mask'], 'xlnet': ['input_ids', 'token_type_ids', 'attention_mask']}

	BATCH_SIZE = 32
	MAX_SEQ_LEN = 128
	LEARNING_RATE = 3e-5
    


	def __init__(self, name):
		self.name = name
		self.classification_model = TFAutoModelForSequenceClassification
		self.qna_model = TFAutoModelForQuestionAnswering
		self.type = self.MODEL_MAP[self.name]
		self.tokenizer = AutoTokenizer.from_pretrained(self.type[0])


	def load_model(self, taskclass, task, num_classes):
		if task == 'sentiment':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((self.MAX_SEQ_LEN,), dtype='int32', name=ip))
			base_model = taskclass.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			y = Activation('softmax')(y)

			model = tf.keras.Model(inputs, y)
			return model
		elif task == 'multilabel':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((self.MAX_SEQ_LEN,), dtype='int32', name=ip))
			base_model = taskclass.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			y = Activation('sigmoid')(y)

			model = tf.keras.Model(inputs, y)
			return model
		elif task == 'qna':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((self.MAX_SEQ_LEN,), dtype='int32', name=ip))
			base_model = taskclass.from_pretrained(self.type[0], output_attentions=False, 
															output_hidden_states=False)
