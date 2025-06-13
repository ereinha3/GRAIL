import json
from collections import defaultdict
from TSP import TSP_Task

tsp = TSP_Task()

hard = [i for i in range(tsp.hard_min_nodes, tsp.hard_max_nodes + 1)]
easy = [i for i in range(tsp.easy_min_nodes, tsp.easy_max_nodes + 1)]
with open('outputs/accuracy.json') as f:
	accuracies = json.load(f)

easy_acc = {}
easy_hall = {}
hard_acc = {}
hard_hall = {}

def make_dict():
	return {'acc': 0, 'feas': 0, 'hall': 0}

def make_hall_dict():
	return {
                "wrong_format": 0,
                "wrong_length": 0,
                "duplicate_nodes": 0,
                "made_up_node": 0,
      	}

for model in accuracies.keys():
	easy_acc[model] = {}
	hard_acc[model] = {}
	easy_hall[model] = {}
	hard_hall[model] = {}
	for encoder in accuracies[model].keys():
		easy_sum_dict = make_dict()
		hard_sum_dict = make_dict()
		easy_hall_dict = make_hall_dict()
		hard_hall_dict = make_hall_dict()
		for size in accuracies[model][encoder].keys():
			sum_dict, hall_dict = (easy_sum_dict, easy_hall_dict) if int(size) in easy else (hard_sum_dict, hard_hall_dict)
			print('easy' if int(size) in easy else 'hard', size)
			for metric in accuracies[model][encoder][size].keys():
				if metric == 'hall':
					sum_dict[metric] += 1 - accuracies[model][encoder][size]['feas'] - accuracies[model][encoder][size]['acc']
					for submetric in accuracies[model][encoder][size][metric].keys():
						hall_dict[submetric] += accuracies[model][encoder][size][metric][submetric]
				elif metric == 'feas':
					sum_dict[metric] += accuracies[model][encoder][size][metric] + accuracies[model][encoder][size]['acc']
				else:
					sum_dict[metric] += accuracies[model][encoder][size][metric]
		for metric in sum_dict.keys():
			easy_sum_dict[metric] = easy_sum_dict[metric] / (len(easy))
			hard_sum_dict[metric] = hard_sum_dict[metric] / (len(hard))
		for metric in hall_dict.keys():
			easy_hall_dict[metric] = easy_hall_dict[metric] / (len(easy))
			hard_hall_dict[metric] = hard_hall_dict[metric] / (len(hard))
		easy_acc[model][encoder] = easy_sum_dict
		hard_acc[model][encoder] = hard_sum_dict
		easy_hall[model][encoder] = easy_hall_dict
		hard_hall[model][encoder] = hard_hall_dict

for model in easy_acc.keys():
	for encoder in easy_acc[model].keys():
		print(model, encoder)
		for dict, name in [(easy_acc, 'easy'), (hard_acc, 'hard'), (easy_hall, 'easy_hall'), (hard_hall, 'hard_hall')]:
			for key in dict[model][encoder].keys():
				print(f'\t{name} {key}: {dict[model][encoder][key]}')
		print()

