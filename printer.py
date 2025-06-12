import json
from collections import defaultdict

hard = [i for i in range(10, 16)]
easy = [i for i in range(4, 10)]
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
	for encoder in accuracies[model].keys():
		easy_sum_dict = make_dict()
		hard_sum_dict = make_dict()
		easy_hall_dict = make_hall_dict()
		hard_hall_dict = make_hall_dict()
		for size in accuracies[model][encoder].keys():
			sum_dict, hall_dict = (easy_sum_dict, easy_hall_dict) if size in easy else (hard_sum_dict, hard_hall_dict)
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
			easy_sum_dict[metric] = sum_dict[metric] / (len(easy))
			hard_sum_dict[metric] = sum_dict[metric] / (len(hard))
		for metric in hall_dict.keys():
			easy_hall_dict[metric] = hall_dict[metric] / (len(easy))
			hard_hall_dict[metric] = hall_dict[metric] / (len(hard))
		easy_acc[model][encoder] = easy_sum_dict
		hard_acc[model][encoder] = hard_sum_dict
		easy_hall[model][encoder] = easy_hall_dict
		hard_hall[model][encoder] = hard_hall_dict

for model in easy_acc.keys():
	print(model)
	for encoder in easy_acc[model].keys():
		print(encoder)
		print(easy_acc[model][encoder])
		print(hard_acc[model][encoder])
		print(easy_hall[model][encoder])
		print(hard_hall[model][encoder])

