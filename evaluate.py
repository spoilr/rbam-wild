import sys
sys.path.insert(0, 'deep/')
sys.path.insert(0, 'relations/')
from deep_training import final_model
from processing import get_carstens_data
from parse_data import NR_TOTAL_ARGS_ANGRY
from parse_data import NR_TOTAL_ARGS_MERGED
from parse_data import NR_TOTAL_ARGS_UNMERGED
from parse_data import pandas_matrix
from parse_data import angry_men_submissions
from parse_data import merged_flat_submissions
from parse_data import unmerged_flat_submissions
from keras.models import load_model
from angry_men_data import angry_men_script
from flat_data import merged_flat_script
from flat_data import unmerged_flat_script
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import base_filter
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from relations_main import get_trained_alg
from similarity_features import Similarity_features
from utils import pos_sentences
from nltk.stem import WordNetLemmatizer
import itertools
import numpy as np
import pickle
import pandas
pandas.set_option('display.expand_frame_repr', False)


MAX_SEQUENCE_LENGTH = 50
NR_CLASSES = 3
NEITHER_CLASS = 0
SUPPORT_CLASS = 1
ATTACK_CLASS = 2

########## DEEP ##########


def tensors(parents, children, tokenizer):
	parents_sequences = tokenizer.texts_to_sequences(parents)
	children_sequences = tokenizer.texts_to_sequences(children)

	parents_tensor = pad_sequences(parents_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	children_tensor = pad_sequences(children_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	
	return parents_tensor, children_tensor


def predict_class_from_tensors(model, parents_tensor, children_tensor):
	predicted_class = model.predict_classes([np.array([parents_tensor]), np.array([children_tensor])], batch_size=1, verbose=0)[0]
	if predicted_class == ATTACK_CLASS:
		return -1
	elif predicted_class == SUPPORT_CLASS:
		return 1
	elif predicted_class == NEITHER_CLASS:
		return 0


def predict_class_from_sentences(model, parent, child, tokenizer):
	parents_tensor, children_tensor = tensors([parent], [child], tokenizer)
	return predict_class_from_tensors(model, parents_tensor[0], children_tensor[0])


########## NON-DEEP ##########


def get_pred(data, model):
	return model.predict([data])

def rf_predict_class(orig_sent1, orig_sent2, model):
	lemmatizer = WordNetLemmatizer()

	sent1 = itertools.chain(*pos_sentences([orig_sent1]))
	sent2 = itertools.chain(*pos_sentences([orig_sent2]))

	sim_features = Similarity_features(orig_sent1, orig_sent2, sent1, sent2, lemmatizer, 0.8)
	vectorized_pair = sim_features.create_vectorized_features()
	return get_pred(vectorized_pair, model)[0]


########## SEARCH BEST ##########


def find_best(text):
	parents, children, all_texts, targets = get_carstens_data()

	tokenizer = pickle.load(open('data_deep_models/deep_tokenizer.pkl', 'r'))

	if text == 'merged':
		submissions_count_matrix = merged_flat_submissions()
	elif text == 'unmerged':
		submissions_count_matrix = unmerged_flat_submissions()
	elif text == 'angry':
		submissions_count_matrix = angry_men_submissions()

	to_continue = True

	model_to_test = raw_input('model_to_test ')

	while to_continue:
		model, _ = final_model(model_to_test, 2, parents, children, all_texts, targets)
		
		if text == 'merged':
			count_matrix = merged_args_rels(model, tokenizer, 'deep')
		if text == 'unmerged':
			count_matrix = unmerged_args_rels(model, tokenizer, 'deep')
		elif text == 'angry':
			count_matrix = angry_men_args_rels(model, tokenizer, 'deep')

		rels_per_6, _, att_per_6, _, supp_per_6, _ = evaluate_model_against_submissions(submissions_count_matrix, count_matrix, 6)

		print "r%d - s%d - a%d" % (rels_per_6, supp_per_6, att_per_6)
		if att_per_6 >= 3 and supp_per_6 >= 4:
			to_continue = False


def test_best():
	submissions_count_matrix = submissions()
	threshold = int(raw_input('threhold agreement '))

	model = '7concat_True_lstm_dropout_after_lstm.h5'
	tokenizer = pickle.load(open('data_deep_models/deep_tokenizer.pkl', 'r'))
	deep_model = load_model(model)
	count_matrix = angry_men_args_rels(deep_model, tokenizer, 'deep')

	print pandas_matrix(submissions_count_matrix)
	print pandas_matrix(count_matrix)
	print evaluate_model_against_submissions(submissions_count_matrix, count_matrix, threshold)


########## EVALUATE ##########

def angry_men_args_rels(model, tokenizer, model_type):
	count_matrix = np.empty(shape=[NR_TOTAL_ARGS_ANGRY, NR_TOTAL_ARGS_ANGRY], dtype=object)
	for i in range(NR_TOTAL_ARGS_ANGRY):
		for j in range(NR_TOTAL_ARGS_ANGRY):
			count_matrix[i][j] = [0, 0]

	for idx_parent in range(NR_TOTAL_ARGS_ANGRY):
		for idx_child in range(idx_parent+1, NR_TOTAL_ARGS_ANGRY):
			if model_type == 'deep':
				predicted_class = predict_class_from_sentences(model, angry_men_script[idx_parent], angry_men_script[idx_child], tokenizer)
			elif model_type == 'rf':
				predicted_class = rf_predict_class(angry_men_script[idx_parent], angry_men_script[idx_child], model)	
			if predicted_class == -1:
				count_matrix[idx_child][idx_parent][0] += 1
			elif predicted_class == 1:
				count_matrix[idx_child][idx_parent][1] += 1

	return count_matrix


def merged_args_rels(model, tokenizer, model_type):
	count_matrix = np.empty(shape=[NR_TOTAL_ARGS_MERGED, NR_TOTAL_ARGS_MERGED], dtype=object)
	for i in range(NR_TOTAL_ARGS_MERGED):
		for j in range(NR_TOTAL_ARGS_MERGED):
			count_matrix[i][j] = [0, 0]

	for idx_parent in range(NR_TOTAL_ARGS_MERGED):
		for idx_child in range(idx_parent+1, NR_TOTAL_ARGS_MERGED):
			if model_type == 'deep':
				predicted_class = predict_class_from_sentences(model, merged_flat_script[idx_parent], merged_flat_script[idx_child], tokenizer)
			elif model_type == 'rf':
				predicted_class = rf_predict_class(merged_flat_script[idx_parent], merged_flat_script[idx_child], model)	
			if predicted_class == -1:
				count_matrix[idx_child][idx_parent][0] += 1
			elif predicted_class == 1:
				count_matrix[idx_child][idx_parent][1] += 1

	return count_matrix

def unmerged_args_rels(model, tokenizer, model_type):
	count_matrix = np.empty(shape=[NR_TOTAL_ARGS_UNMERGED, NR_TOTAL_ARGS_UNMERGED], dtype=object)
	for i in range(NR_TOTAL_ARGS_UNMERGED):
		for j in range(NR_TOTAL_ARGS_UNMERGED):
			count_matrix[i][j] = [0, 0]

	for idx_parent in range(NR_TOTAL_ARGS_UNMERGED):
		for idx_child in range(NR_TOTAL_ARGS_UNMERGED):
			if model_type == 'deep':
				predicted_class = predict_class_from_sentences(model, unmerged_flat_script[idx_parent], unmerged_flat_script[idx_child], tokenizer)
			elif model_type == 'rf':
				predicted_class = rf_predict_class(unmerged_flat_script[idx_parent], unmerged_flat_script[idx_child], model)	
			if predicted_class == -1:
				count_matrix[idx_child][idx_parent][0] += 1
			elif predicted_class == 1:
				count_matrix[idx_child][idx_parent][1] += 1

	return count_matrix

def evaluate_model_against_submissions(submissions_count_matrix, count_matrix, threshold):
	nr_total_rels = 0
	nr_rels = 0
	nr_total_att = 0
	nr_att = 0
	nr_total_supp = 0
	nr_supp = 0
	for i in range(len(submissions_count_matrix)):
		for j in range(len(submissions_count_matrix)):
			submissions_cell_data = submissions_count_matrix[i][j]
			cell_data = count_matrix[i][j]
			if not (submissions_cell_data[0] == 0 and submissions_cell_data[1] == 0):
				maximum_agreement = max(submissions_cell_data)
				if maximum_agreement >= threshold:
					nr_total_rels += 1

					rel = submissions_cell_data.index(maximum_agreement)
					if rel == 0:
						nr_total_att += 1
					if rel == 1:
						nr_total_supp += 1

					if cell_data[rel] == 1:
						nr_rels += 1

					if rel == 0 and cell_data[rel] == 1:
						nr_att += 1

					if rel == 1 and cell_data[rel] == 1:
						nr_supp += 1

	if nr_total_supp == 0:
		nr_supp_percentage = 0
	else:
		nr_supp_percentage = nr_supp*100/nr_total_supp
	
	if nr_total_att == 0:
		nr_att_percentage = 0
	else:
		nr_att_percentage = nr_att*100/nr_total_att

	return nr_rels, nr_rels*100/nr_total_rels, nr_att, nr_att_percentage, nr_supp, nr_supp_percentage


def eval_angry(model, tokenizer, model_type):
	submissions_count_matrix = angry_men_submissions()
	print pandas_matrix(submissions_count_matrix, 'angry')

	count_matrix = angry_men_args_rels(model, tokenizer, model_type)
	print pandas_matrix(count_matrix, 'angry')

	idx = [6,9,10,11,12,13,15,18,20]
	for i in idx:
		print evaluate_model_against_submissions(submissions_count_matrix, count_matrix, i)

def eval_merged(model, tokenizer, model_type):
	submissions_count_matrix = merged_flat_submissions()
	print pandas_matrix(submissions_count_matrix, 'merged')

	count_matrix = merged_args_rels(model, tokenizer, model_type)
	print pandas_matrix(count_matrix, 'merged')

	idx = [8,9,14,16,19,20]
	for i in idx:
		print evaluate_model_against_submissions(submissions_count_matrix, count_matrix, i)

def eval_unmerged(model, tokenizer, model_type):
	submissions_count_matrix = unmerged_flat_submissions()
	print pandas_matrix(submissions_count_matrix, 'unmerged')

	count_matrix = unmerged_args_rels(model, tokenizer, model_type)
	print pandas_matrix(count_matrix, 'unmerged')

	idx = [8,9,14,16,19,20]
	for i in idx:
		print evaluate_model_against_submissions(submissions_count_matrix, count_matrix, i)

def test_best_diff_thresholds():
	model_type = raw_input('model_type ')
	tokenizer = pickle.load(open('data_deep_models/deep_tokenizer.pkl', 'r'))

	if model_type == 'deep':
		model = load_model('7concat_True_lstm_dropout_after_lstm.h5')
	elif model_type == 'rf':
		model = get_trained_alg()

	text = raw_input('merged or angry ')
	if text == 'merged':
		eval_merged(model, tokenizer, model_type)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_MERGED
	elif text == 'unmerged':
		eval_unmerged(model, tokenizer, model_type)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_MERGED
	elif text == 'angry':
		eval_angry(model, tokenizer, model_type)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_ANGRY



if __name__ == '__main__':
	text = raw_input('merged or angry ')
	# find_best(text)
	# test_best()

	test_best_diff_thresholds()

