import copy
import json
import tqdm
from solutions_evaluation.abstract_solver import AbstractSolver
from solution1_few_shot_with_python.solver import Solver1
from time import time
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

results = list(map(json.loads, open('test_examples_3solution1_results.jsonl')))
data = pd.DataFrame(results)
data['cur_digits'] = data['cur_digits'].fillna(0)
data['max_digits'] = data['max_digits'].fillna(0)

data['cur_digits'] = data['cur_digits'] + data['max_digits']
data['a_neg'] = (data['a'] < 0).astype(int)
data['b_neg'] = (data['b'] < 0).astype(int)
data['answer_neg'] = (pd.to_numeric(data['answer_int']) < 0).astype(int)
data['target_neg'] = (data['target'] < 0).astype(int)
data['score'] = (data['answer_int'] == data['target']).astype(int)
data['answer_was_given'] = (data['answer_int'] != '').astype(int)

data = data.drop(['max_digits', ], axis=1)
data = data.drop(['target', 'a', 'b', 'answer_int', ], axis=1)


print(data)
print(data.columns)

print('All errors')
print(data[data['score'] == 0])

print('When signs are different')
print(data[data['answer_neg'] != data['target_neg']])
