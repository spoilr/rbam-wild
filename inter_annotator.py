from angry_men_data import *
from parse_data import pandas_matrix
from parse_data import angry_men_submissions
from parse_data import merged_flat_submissions
from parse_data import unmerged_flat_submissions
from parse_data import create_tuples_dict
from parse_data import create_tuples_dict_with_model
from parse_data import get_tuple
import numpy as np
import pandas
pandas.set_option('display.expand_frame_repr', False)



def create_file_for_model(string, rels):
	file_name = 'model.txt'

	attacks = [x[x.find("(")+1:x.find(")")] for x in model if 'attacks' in x]
	supports = [x[x.find("(")+1:x.find(")")] for x in model if 'supports' in x]

	attacks = [get_tuple(x) for x in attacks]
	supports = [get_tuple(x) for x in supports]

	for idx, rel in enumerate(rels):
		with open(file_name, "a") as myfile:
			myfile.write(str(idx) + ' ')
		if rel in attacks:
			with open(file_name, "a") as myfile:
				myfile.write('a\n')
		elif rel in supports:
			with open(file_name, "a") as myfile:
				myfile.write('s\n')
		else:
			with open(file_name, "a") as myfile:
				myfile.write('n\n')


def inter_annotator_baf(tuples_dict, model):
	if model == 'angry':
		submissions_count_matrix = angry_men_submissions()
	elif model == 'merged':
		submissions_count_matrix = merged_flat_submissions()
	elif model == 'unmerged':
		submissions_count_matrix = unmerged_flat_submissions()
	rels = []

	for i in range(len(submissions_count_matrix)):
		for j in range(len(submissions_count_matrix)):
			if submissions_count_matrix[i][j] != [0, 0]:
				if model == 'angry':
					rels.append(('a'+str(i), 'a'+str(j)))
				elif model == 'merged':
					rels.append(('b'+str(i), 'b'+str(j)))
				elif model == 'unmerged':
					rels.append(('c'+str(i), 'c'+str(j)))

	file_name =  model + '_baf.txt'

	keys = tuples_dict.keys()

	for key in keys:
		with open(file_name, "a") as myfile:
			myfile.write(key + ' ')

	with open(file_name, "a") as myfile:
			myfile.write('\n')

	for rel in rels:
		for key in keys:
			if rel in tuples_dict[key]['attacks']:
				with open(file_name, "a") as myfile:
					myfile.write('a ')
			elif rel in tuples_dict[key]['supports']:
				with open(file_name, "a") as myfile:
					myfile.write('s ')
			else:
				with open(file_name, "a") as myfile:
					myfile.write('n ')
		with open(file_name, "a") as myfile:
			myfile.write('\n')


if __name__ == "__main__":
	text = raw_input('merged or angry: ')
	if text == 'angry':
		tuples_dict = create_tuples_dict_with_model()
		print tuples_dict
		inter_annotator_baf(tuples_dict, 'angry')
	elif text == 'merged':
		tuples_dict = create_tuples_dict('merged')
		print tuples_dict
		inter_annotator_baf(tuples_dict, 'merged')
	elif text == 'unmerged':
		tuples_dict = create_tuples_dict('unmerged')
		print tuples_dict
		inter_annotator_baf(tuples_dict, 'unmerged')

