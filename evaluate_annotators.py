from angry_men_data import angry_men
from angry_men_data import model
from flat_data import merged_flat
from flat_data import unmerged_flat


def create_dict_without_annotator(model, evaluated_annotator):
	rels_dict = {}
	for annotator, rels in model.iteritems():
		if annotator != evaluated_annotator:
			for rel in rels:
				if rel not in rels_dict:
					rels_dict[rel] = 1
				else:
					rels_dict[rel] += 1
	rels_dict['supports(a17,a16)'] = 0
	return rels_dict


def assertion(angry_men_with_model):
	rels_dict = {}
	for annotator, rels in angry_men_with_model.iteritems():
		for rel in rels:
			if rel not in rels_dict:
				rels_dict[rel] = 1
			else:
				rels_dict[rel] += 1

	assert len(rels_dict.keys()) == 73


def filter_out_threshold(annotator_dict, threshold):
	filtered_annotator_dict = {}
	for key in annotator_dict.keys():
		if annotator_dict[key] >= threshold:
			filtered_annotator_dict[key] = annotator_dict[key]
	return filtered_annotator_dict


def evaluate_annotator(filtered_rels_dict, rels):
	rels_total = len(filtered_rels_dict.keys())
	rels_identified = 0
	for rel in rels:
		if rel in filtered_rels_dict:
			rels_identified += 1
	return rels_identified*100/rels_total

def evaluate_annotators_angry():
	angry_men_with_model = angry_men
	angry_men_with_model['model'] = model
	assert len(angry_men_with_model.keys()) == 22
	assertion(angry_men_with_model)

	for annotator, rels in angry_men_with_model.iteritems():
		rels_dict = create_dict_without_annotator(angry_men, annotator)
		filtered_rels_dict = filter_out_threshold(rels_dict, 6)
		score = evaluate_annotator(filtered_rels_dict, rels)
		print (annotator, score)


def evaluate_annotators_merged():
	for annotator, rels in merged_flat.iteritems():
		rels_dict = create_dict_without_annotator(merged_flat, annotator)
		filtered_rels_dict = filter_out_threshold(rels_dict, 6)
		score = evaluate_annotator(filtered_rels_dict, rels)
		print (annotator, score)

def evaluate_annotators_unmerged():
	for annotator, rels in unmerged_flat.iteritems():
		rels_dict = create_dict_without_annotator(unmerged_flat, annotator)
		filtered_rels_dict = filter_out_threshold(rels_dict, 6)
		score = evaluate_annotator(filtered_rels_dict, rels)
		print (annotator, score)

# evaluate_annotators_angry()
# evaluate_annotators_merged()
evaluate_annotators_unmerged()

