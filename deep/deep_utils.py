from embeddings import MAX_SEQUENCE_LENGTH
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import base_filter
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np


NR_CLASSES = 3


def get_sensitive_length_tensors(parents_seq, children_seq, targets, max_length):
	assert len(parents_seq) == len(children_seq)
	if max_length == None:
		parents_sequences = parents_seq
		children_sequences = children_seq
		labels = targets
	else:
		parents_sequences = []
		children_sequences = []
		labels = []
		for i in range(len(parents_seq)):
			if len(parents_seq[i]) <= 50 and len(children_seq[i]) <= 50:
				parents_sequences.append(parents_seq[i])
				children_sequences.append(children_seq[i])
				labels.append(targets[i])

	return parents_sequences, children_sequences, labels


def tokenization(all_texts):
	tokenizer = Tokenizer(filters=base_filter(), lower=True, split=" ")
	tokenizer.fit_on_texts(all_texts)

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	return tokenizer, word_index


def tensors(parents, children, targets, tokenizer):
	parents_sequences = tokenizer.texts_to_sequences(parents)
	children_sequences = tokenizer.texts_to_sequences(children)
	parents_sequences, children_sequences, targets = get_sensitive_length_tensors(parents_sequences, children_sequences, targets, MAX_SEQUENCE_LENGTH)

	parents_tensor = pad_sequences(parents_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	children_tensor = pad_sequences(children_sequences, maxlen=MAX_SEQUENCE_LENGTH)

	labels = to_categorical(np.asarray(targets), NR_CLASSES)

	# all_sequences = parents_sequences + children_sequences
	# print 'max text length %d' % len(max(all_sequences,key=len))
	# print 'min text length %d' % len(min(all_sequences,key=len))
	
	return parents_tensor, children_tensor, labels, targets


def get_sequences(parents, children, all_texts, targets):
	tokenizer, word_index = tokenization(all_texts)
	parents_tensor, children_tensor, labels, targets = tensors(parents, children, targets, tokenizer)

	print('Shape of parents data tensor:', parents_tensor.shape)
	print('Shape of children data tensor:', children_tensor.shape)
	print('Shape of label tensor:', labels.shape)

	return parents_tensor, children_tensor, labels, word_index, targets

