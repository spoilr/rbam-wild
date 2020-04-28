import sys
sys.path.insert(0, 'deep/')
import numpy as np
np.random.seed(1337)
from embeddings import get_embeddings_weights
from embeddings import pretrained_embedding
from embeddings import train_embedding
from embeddings import EMBEDDING_DIM
from embeddings import MAX_SEQUENCE_LENGTH
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge
from keras.models import load_model
from keras.models import Sequential


NR_EPOCHS = 30
BATCH_SIZE = 128
DROPOUT = 0.2
DENSE_SIZE = 32
LSTM_SIZE = 32
ADAM = "adam"


def load_model_from_file(model_file):
	model = load_model(model_file)
	return model


def lstm(word_index, merge_mode, dense_layer_after_merge):
	file_name = merge_mode + "_" + str(dense_layer_after_merge) + '_lstm_dropout_after_lstm.h5'

	embedding_matrix = get_embeddings_weights(word_index)
	
	parent_model = Sequential()
	parent_model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
	
	parent_model.add(LSTM(LSTM_SIZE, activation='relu'))
	parent_model.add(Dropout(DROPOUT))

	children_model = Sequential()
	children_model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
	
	children_model.add(LSTM(LSTM_SIZE, activation='relu'))
	children_model.add(Dropout(DROPOUT))

	model = Sequential()
	model.add(Merge([parent_model, children_model], mode=merge_mode))

	if dense_layer_after_merge:
		model.add(Dense(DENSE_SIZE, activation='relu'))

	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
	
	return model, file_name


def bidir_lstm(word_index, merge_mode, dense_layer_after_merge):
	file_name = merge_mode + "_" + str(dense_layer_after_merge) + '_lstm_dropout_after_lstm_bidir.h5'

	embedding_matrix = get_embeddings_weights(word_index)
	
	parent_model = Sequential()
	parent_model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
	
	parent_model.add(Bidirectional(LSTM(LSTM_SIZE, activation='relu')))
	parent_model.add(Dropout(DROPOUT))

	children_model = Sequential()
	children_model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
	
	children_model.add(Bidirectional(LSTM(LSTM_SIZE, activation='relu')))
	children_model.add(Dropout(DROPOUT))

	model = Sequential()
	model.add(Merge([parent_model, children_model], mode=merge_mode))

	if dense_layer_after_merge:
		model.add(Dense(DENSE_SIZE, activation='relu'))

	model.add(Dense(3, activation='softmax'))
	model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

	return model, file_name


def parse_model_type(model_type, parents_x_train, children_x_train, y_train, word_index):
	if  model_type == 'no_extra_concat_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='concat', dense_layer_after_merge=False)
	elif  model_type == 'no_extra_sum_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='sum', dense_layer_after_merge=False)
	elif  model_type == 'concat_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='concat', dense_layer_after_merge=True)
	elif  model_type == 'sum_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='sum', dense_layer_after_merge=True)
	elif  model_type == 'no_extra_concat_bidir_lstm_dropout_after_lstm':
		model, file_name = bidir_lstm(word_index, merge_mode='concat', dense_layer_after_merge=False)
	elif  model_type == 'no_extra_sum_bidir_lstm_dropout_after_lstm':
		model, file_name = bidir_lstm(word_index, merge_mode='sum', dense_layer_after_merge=False)
	elif  model_type == 'concat_bidir_lstm_dropout_after_lstm':
		model, file_name = bidir_lstm(word_index, merge_mode='concat', dense_layer_after_merge=True)
	elif  model_type == 'sum_bidir_lstm_dropout_after_lstm':
		model, file_name = bidir_lstm(word_index, merge_mode='sum', dense_layer_after_merge=True)

	early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=0)
	checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')	
	model.fit([parents_x_train, children_x_train], y_train, validation_split=0.2, nb_epoch=NR_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, checkpoint], shuffle=True, verbose=0)	
	return model, file_name


