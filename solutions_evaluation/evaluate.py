import copy
import json
import tqdm
from solutions_evaluation.abstract_solver import AbstractSolver
from solution1_few_shot_with_python.solver import Solver1
from time import time

# test_filename = 'test_examples.jsonl'

def evaluate(test_filename: str, output_filename_postfix: str, solverClass, continue_experiment: bool = False):
    output_filename = test_filename[:test_filename.find('.')] + \
                      output_filename_postfix + \
                      test_filename[test_filename.find('.'):]

    start_init = time()
    solver: AbstractSolver = solverClass()
    end_init = time()
    print(f'Class initialization_time: {end_init - start_init}')

    test_examples = list(map(json.loads, open(test_filename)))
    # test_examples = test_examples[:2]

    if continue_experiment:
        lines = open(output_filename).readlines()
        num_skip_exps = len(lines)
        scores = [x['score'] for x in map(json.loads, lines)]
    else:
        num_skip_exps = 0
        scores = []

    with open(output_filename, 'a' if continue_experiment else 'w') as f:
        pbar = tqdm.tqdm(test_examples[num_skip_exps:], initial=num_skip_exps, total=len(test_examples))

        for test_example in pbar:
            a, b = test_example['a'], test_example['b']

            try:
                start_predict = time()
                answer_int, meta_info = solver.calc_sum(a, b)
                end_predict = time()
                answer_int = int(answer_int)
                score = 1 if answer_int == test_example['target'] else 0

            except Exception as e:
                print(e)
                answer_int = ''
                score = 0
            scores.append(score)

            result = copy.copy(test_example)
            result['answer_int'] = answer_int
            result['score'] = score
            result['prediction_time'] = end_predict - start_predict
            result['meta_info'] = meta_info
            f.write(json.dumps(result) + '\n')

            f.flush()

            print(result)
            print(f'Current accuracy - {sum(scores) / len(scores)}')


evaluate('test_examples.jsonl', 'solution1_results_', Solver1, continue_experiment=True)
