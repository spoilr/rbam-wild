from processing import NEITHER_CLASS
from processing import SUPPORT_CLASS
from processing import ATTACK_CLASS
from deep_utils import get_sequences
from deep_utils import tensors
from models import parse_model_type
from models import load_model_from_file
from models import lstm
from models import bidir_lstm
from models import NR_EPOCHS
from models import BATCH_SIZE
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import pickle
import numpy as np


NR_FOLDS = 10
VALIDATION_SPLIT = 0.2



def predict(parents, children, y, model_file):
	model = load_model_from_file(model_file)
	scores = model.evaluate([parents, children], y, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
	print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
	print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))

	return scores[1]*100


def final_model(model_type, nr_epochs, parents, children, all_texts, targets):
	parents_tensor, children_tensor, labels, word_index, targets = get_sequences(parents, children, all_texts, targets)

	if  model_type == 'no_extra_concat_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='concat', dense_layer_after_merge=False)
	elif  model_type == 'concat_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='concat', dense_layer_after_merge=True)
	elif  model_type == 'sum_lstm_dropout_after_lstm':
		model, file_name = lstm(word_index, merge_mode='sum', dense_layer_after_merge=True)
	elif  model_type == 'concat_bidir_lstm_dropout_after_lstm':
		model, file_name = bidir_lstm(word_index, merge_mode='concat', dense_layer_after_merge=True)

	early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=0)
	checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')	
	model.fit([parents_tensor, children_tensor], labels, validation_split=VALIDATION_SPLIT, nb_epoch=nr_epochs, batch_size=BATCH_SIZE, callbacks=[early_stopping, checkpoint], shuffle=False, verbose=0)

	predict(parents_tensor, children_tensor, labels, file_name)

	return model, file_name

	
