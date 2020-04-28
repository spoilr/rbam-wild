import os
from keras.layers import Embedding
import numpy as np

GLOVE_DIR = 'deep/glove.6B'
GLOVE_FILE = 'glove.6B.100d.txt'

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 50

def get_embeddings_index():
	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	#print('Found %s word vectors.' % len(embeddings_index))
	return embeddings_index


def get_embeddings(word_index, embeddings_index):
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


def get_embeddings_weights(word_index):
	embeddings_index = get_embeddings_index()
	embedding_matrix = get_embeddings(word_index, embeddings_index)
	return embedding_matrix


def pretrained_embedding(word_index):
	embedding_matrix = get_embeddings_weights(word_index)
	
	embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

	return embedding_layer


def train_embedding(word_index):
	embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

	return embedding_layer	
