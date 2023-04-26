import random
import json

random.seed(0)


def go(filename='test_examples.jsonl', n_reapeats=50):
    min_e = 1e20
    max_e = 1e30
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




# Random sampling: sample [number1] and [number2] independently from [0, 10^D −1]
def random_sampling(n_numbers: int = 100, max_d: int = 100, filename='test_examples.jsonl', ):
    max_number = pow(10, max_d) - 1

    with open(filename, 'w') as f:
        for i in range(n_numbers):
            a = random.randint(0, max_number)
            b = random.randint(0, max_number)
            example = {
                'a': a,
                'b': b,
                'target': a + b
            }

            f.write(json.dumps(example) + '\n')
            f.flush()


def sample_with_step(filename='test_examples.jsonl', n_each=1, min_d=1, max_d=101, step_d=10):
    for cur_d in range(min_d, max_d + 1, step_d):
        max_number = pow(10, cur_d) - 1

        #print(cur_d, min_number, max_number)

        for min_a_digits, min_b_digits in [[cur_d, cur_d], [1, cur_d], [cur_d, 1]]:
            min_a_number = pow(10, min_a_digits - 1)
            min_b_number = pow(10, min_b_digits - 1)
            for _ in range(n_each):
                a = random.randint(min_a_number, max_number) * random.choice([-1, 1])
                b = random.randint(min_b_number, max_number) * random.choice([-1, 1])
                example = {
                    'a': a,
                    'b': b,
                    'target': a + b,
                    'max_digits': cur_d,
                    'min_a_digits': min_a_digits,
                    'min_b_digits': min_b_digits
                }

                with open(filename, 'w') as f:
                    f.write(json.dumps(example) + '\n')
                    f.flush()

                print(example, 'a_len', len(str(abs(a))), 'b_len', len(str(abs(b))))


# go()
#random_sampling()
sample_with_step()
