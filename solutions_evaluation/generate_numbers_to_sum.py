import random
import json
from math import floor, sqrt

random.seed(0)


def sample_with_step(filename='test_examples_3.jsonl', n_each=3, min_d=10, max_d=101, step_d=10):
    with open(filename, 'w'):
        pass

    for cur_d in range(min_d, max_d + 1, step_d):
        max_number = pow(10, cur_d) - 1
        min_number = pow(10, cur_d - 1)

        # print(cur_d, min_number, max_number, floor(sqrt(max_number)))

        # for min_a_digits, min_b_digits in [[cur_d, cur_d], [1, cur_d], [cur_d, 1]]:
        #     min_a_number = pow(10, min_a_digits - 1)
        #     min_b_number = pow(10, min_b_digits - 1)

        for _ in range(n_each):
            a_equal = random.randint(min_number, max_number) * random.choice([-1, 1])
            b_equal = random.randint(min_number, max_number) * random.choice([-1, 1])

            example_equal = {
                'a': a_equal,
                'b': b_equal,
                'target': a_equal + b_equal,
                'max_digits': cur_d,
                'a_digits': len(str(abs(a_equal))),
                'b_digits': len(str(abs(b_equal))),
            }

            a_not_equal = random.randint(0, floor(sqrt(max_number))) * random.choice([-1, 1])
            b_not_equal = random.randint(min_number, max_number) * random.choice([-1, 1])

            a_b = [a_not_equal, b_not_equal]
            random.shuffle(a_b)
            a_not_equal, b_not_equal = a_b

            example_not_equal = {
                'a': a_not_equal,
                'b': b_not_equal,
                'target': a_not_equal + b_not_equal,
                'cur_digits': cur_d,
                'a_digits': len(str(abs(a_not_equal))),
                'b_digits': len(str(abs(b_not_equal))),
            }


            with open(filename, 'a') as f:
                f.write(json.dumps(example_equal) + '\n')
                f.write(json.dumps(example_not_equal) + '\n')

                f.flush()

            print(example_equal)
            print(example_not_equal)

if __name__ == '__main__':
    sample_with_step()
