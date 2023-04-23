import copy
import json
import argparse
import tqdm
import os
import openai
from interface import MyProgramInterface
from prompts_formatting import generate_prompt

openai.api_key = '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'

SUM_PROMPT = generate_prompt()


class Settings:
    def __init__(self, model_name, test_examples_path=f'test_examples.jsonl', results_path='eval_results/my_first_results.jsonl',
                 continue_experiment=False, temperature=0.0, top_p=1.0, max_tokens=128, majority_at=None):
        self.model_name = model_name
        self.test_examples_path = test_examples_path
        self.results_path = results_path

        self.continue_experiment = continue_experiment

        # OpenAI recommends using either temperature or top_p, but not both.

        #  higher values like 0.8 will make the output more random,
        #  while lower values like 0.2 will make it more focused and deterministic.
        self.temperature = temperature

        # Top P or top-k sampling or nucleus sampling is a parameter to control the
        # sampling pool from the output distribution.
        # For example, value 0.1 means the model only samples the output from the top 10% of the distribution.
        # The value range was between 0 and 1; higher values mean a more diverse result.
        self.top_p = top_p

        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens = max_tokens

        self.majority_at = majority_at  # todo??


# 'text-davinci-003', 'gpt-3.5-turbo'
MODEL = 'gpt-3.5-turbo'
settings = Settings(MODEL)


os.makedirs(os.path.dirname(settings.results_path), exist_ok=True)
test_examples = list(map(json.loads, open(settings.test_examples_path)))
test_examples = test_examples[:2]

interface = MyProgramInterface(stop='\n\n\n', get_answer_expr='solution()', model=settings.model_name, verbose=True)

if settings.continue_experiment:
    lines = open(settings.results_path).readlines()
    num_skip_exps = len(lines)
    scores = [x['score'] for x in map(json.loads, lines)]
else:
    num_skip_exps = 0
    scores = []

with open(settings.results_path, 'a' if settings.continue_experiment else 'w') as f:
    pbar = tqdm.tqdm(test_examples[num_skip_exps:], initial=num_skip_exps, total=len(test_examples))

    for test_example in pbar:
        a, b = test_example['a'], test_example['b']
        question = SUM_PROMPT.format(a=a, b=b)

        try:
            ans = interface.run(question, majority_at=settings.majority_at,
                                temperature=settings.temperature,
                                max_tokens=settings.max_tokens)
            ans = float(ans)
            score = 1 if abs(ans - test_example['target']) < 1e-3 else 0

        except Exception as e:
            print(e)
            ans = ''
            score = 0
        scores.append(score)

        result = copy.copy(test_example)
        result['answer'] = ans
        result['score'] = score
        result['generation'] = interface.history
        f.write(json.dumps(result) + '\n')

        interface.clear_history()
        f.flush()

print(f'Accuracy - {sum(scores) / len(scores)}')
