import sys
sys.path.insert(0, 'carstens_corpora/')
sys.path.insert(0, 'data/')
from xml.dom import minidom
from carstens_load_data import load_carstens_data

NEITHER_CLASS = 0
SUPPORT_CLASS = 1
ATTACK_CLASS = 2

def process_targets(targets):
	processed_targets = []
	for x in targets:
		if x == -1:
			processed_targets.append(ATTACK_CLASS)
		elif x == 1:
			processed_targets.append(SUPPORT_CLASS)
		elif x == 0:
			processed_targets.append(NEITHER_CLASS)	
	return processed_targets


def get_carstens_data():
	sentences, targets = load_carstens_data()
	targets = process_targets(targets)

	assert len(sentences) == len(targets)
	assert len(sentences) == 16857

	print 'attackers %d' % len([x for x in targets if x == ATTACK_CLASS])
	print 'supporters %d' % len([x for x in targets if x == SUPPORT_CLASS])
	print 'neither %d' % len([x for x in targets if x == NEITHER_CLASS])

	parents = [x[0].encode('ascii', 'ignore') for x in sentences]
	children = [x[1].encode('ascii', 'ignore') for x in sentences]
	all_texts = parents + children

	assert len(parents) == len(children) == len(targets)

	return parents, children, all_texts, targets


