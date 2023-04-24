import random
import json

random.seed(0)

min_e = 1e20
max_e = 1e30
def go(filename='test_examples.jsonl', n_reapeats=50):
    with open(filename, 'w') as f:
        for i in range(n_reapeats):
            a = random.randint(int(min_e), int(max_e))
            b = random.randint(int(min_e), int(max_e))
            example = {
                'a': a,
                'b': b,
                'target': a + b
            }

            f.write(json.dumps(example) + '\n')
            f.flush()

go()