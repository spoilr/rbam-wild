from angry_men_data import angry_men
from angry_men_data import model
from flat_data import merged_flat
from flat_data import unmerged_flat
import re
import numpy as np
import pandas
pandas.set_option('display.expand_frame_repr', False)


NR_TOTAL_ARGS_ANGRY = 18
NR_TOTAL_ARGS_MERGED = 12
NR_TOTAL_ARGS_UNMERGED = 16
ATTACKS = 'attacks'
SUPPORTS = 'supports'


def get_tuple(string_tuple):
	tuple_data = re.split(', *', string_tuple)
	return (tuple_data[0], tuple_data[1])


def create_tuples_dict_with_model():
	tuples_dict = create_tuples_dict('angry')

	attacks = [x[x.find("(")+1:x.find(")")] for x in model if 'attacks' in x]
	supports = [x[x.find("(")+1:x.find(")")] for x in model if 'supports' in x]

	tuples_dict['model'] = dict()
	tuples_dict['model'][ATTACKS] = [get_tuple(x) for x in attacks]
	tuples_dict['model'][SUPPORTS] = [get_tuple(x) for x in supports]
	
	return tuples_dict

def create_tuples_dict(model):
	tuples_dict = dict()
	
	if model == 'angry':
		data_dictionary = angry_men
	elif model == 'merged':
		data_dictionary = merged_flat
	elif model == 'unmerged':
		data_dictionary = unmerged_flat

	for uid, baf in data_dictionary.iteritems():
		attacks = [x[x.find("(")+1:x.find(")")] for x in baf if 'attacks' in x]
		supports = [x[x.find("(")+1:x.find(")")] for x in baf if 'supports' in x]

		assert len(attacks) + len(supports) == len(baf)

		tuples_dict[uid] = dict()
		tuples_dict[uid][ATTACKS] = [get_tuple(x) for x in attacks]
		tuples_dict[uid][SUPPORTS] = [get_tuple(x) for x in supports]

	return tuples_dict


# (nr_attacks, nr_supports)
def create_count_matrix(tuples_dict, text):
	if text == 'angry':
		count_matrix = np.empty(shape=[NR_TOTAL_ARGS_ANGRY, NR_TOTAL_ARGS_ANGRY], dtype=object)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_ANGRY
	elif text == 'merged':
		count_matrix = np.empty(shape=[NR_TOTAL_ARGS_MERGED, NR_TOTAL_ARGS_MERGED], dtype=object)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_MERGED
	elif text == 'unmerged':
		count_matrix = np.empty(shape=[NR_TOTAL_ARGS_UNMERGED, NR_TOTAL_ARGS_UNMERGED], dtype=object)
		NR_TOTAL_ARGS = NR_TOTAL_ARGS_UNMERGED

	for i in range(NR_TOTAL_ARGS):
		for j in range(NR_TOTAL_ARGS):
			count_matrix[i][j] = [0, 0]

	for k, rels in tuples_dict.iteritems():
		for (c, p) in rels['attacks']:
			if text == 'angry':
				count_matrix[int(c.split('a')[1])][int(p.split('a')[1])][0] += 1
			elif text == 'merged':
				count_matrix[int(c.split('b')[1])][int(p.split('b')[1])][0] += 1
			elif text == 'unmerged':
				count_matrix[int(c.split('c')[1])][int(p.split('c')[1])][0] += 1
		for (c, p) in rels['supports']:
			if text == 'angry':
				count_matrix[int(c.split('a')[1])][int(p.split('a')[1])][1] += 1
			elif text == 'merged':
				count_matrix[int(c.split('b')[1])][int(p.split('b')[1])][1] += 1
			elif text == 'unmerged':
				count_matrix[int(c.split('c')[1])][int(p.split('c')[1])][1] += 1

	return count_matrix


def add_model_to_count_matrix(count_matrix):
	attacks = [x[x.find("(")+1:x.find(")")] for x in model if 'attacks' in x]
	supports = [x[x.find("(")+1:x.find(")")] for x in model if 'supports' in x]

	attacks = [get_tuple(x) for x in attacks]
	supports = [get_tuple(x) for x in supports]

	for (c, p) in attacks:
		count_matrix[int(c.split('a')[1])][int(p.split('a')[1])][0] += 1
	for (c, p) in supports:
		count_matrix[int(c.split('a')[1])][int(p.split('a')[1])][1] += 1

	return count_matrix


def pandas_matrix(count_matrix, text):
	if text == 'angry':
		labels = ['a%d' % i for i in range(NR_TOTAL_ARGS_ANGRY)]
	elif text == 'merged':
		labels = ['b%d' % i for i in range(NR_TOTAL_ARGS_MERGED)]
	elif text == 'unmerged':
		labels = ['c%d' % i for i in range(NR_TOTAL_ARGS_UNMERGED)]
	df = pandas.DataFrame(count_matrix, columns=labels, index=labels)
	df = df.astype(str)
	df.replace('[0, 0]', '', inplace=True)
	return df

def angry_men_submissions():
	tuples_dict = create_tuples_dict('angry')
	count_matrix = create_count_matrix(tuples_dict, 'angry')
	count_matrix = add_model_to_count_matrix(count_matrix)
	return count_matrix


def merged_flat_submissions():
	tuples_dict = create_tuples_dict('merged')
	count_matrix = create_count_matrix(tuples_dict, 'merged')
	return count_matrix

def unmerged_flat_submissions():
	tuples_dict = create_tuples_dict('unmerged')
	count_matrix = create_count_matrix(tuples_dict, 'unmerged')
	return count_matrix

if __name__ == "__main__":
	text = raw_input('merged or angry ')
	if text == 'merged':
		count_matrix = merged_flat_submissions()
	elif text == 'unmerged':
		count_matrix = unmerged_flat_submissions()
	elif text == 'angry':
		count_matrix = angry_men_submissions()
	df = pandas_matrix(count_matrix, text)
	print df