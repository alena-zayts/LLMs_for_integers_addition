import random
import json
import random
random.seed(0)
from prompts_formatting import code_for_one_task, header_for_one_task, full_question_with_code_for_one_task

def go(filename='my_test.jsonl', n_reapeats=50):
    with open(filename, 'w') as f:
        for i in range(n_reapeats):
            a = random.randint(-10000000, 1000000)
            b = random.randint(-10000000, 1000000)
            example = {
                'a': a,
                'b': b,
                # 'input': header_for_one_task(a, b),
                # 'code': code_for_one_task(a, b),
                'target': a + b
            }

            f.write(json.dumps(example) + '\n')
            f.flush()

go()