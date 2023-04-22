import copy
import json
import argparse
import tqdm
import os

from interface import MyProgramInterface
from prompt import MATH_PROMPT

import openai
openai.api_key = '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'



parser = argparse.ArgumentParser()
args = parser.parse_args()
args.dataset = 'gsm'  # TODO: здесь обучающие вопросы
args.append = False  # добавляются ли данные или перезаписываются
args.temperature = 0.0
# Top P or top-k sampling or nucleus sampling is a parameter to control the sampling pool from the output distribution.
# For example, value 0.1 means the model only samples the output from the top 10% of the distribution.
# The value range was between 0 and 1; higher values mean a more diverse result.
args.top_p = 1.0  # TODO
# The maximum number of tokens to generate in the chat completion.
args.max_tokens = 256  # TOD
args.majority_at = None  # todo??


DATA_PATH = f'gsmhardv2.jsonl'
OUTPUT_PATH = f'eval_results/{args.dataset}.jsonl'
MODEL = 'text-davinci-003'
MODEL = 'gpt-3.5-turbo'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

examples = list(map(json.loads, open(DATA_PATH)))
examples = examples[:10]
itf = MyProgramInterface(
    stop='\n\n\n',
    get_answer_expr='solution()',
    model=MODEL,
    verbose=True
)

if args.append:
    lines = open(OUTPUT_PATH).readlines()
    num_skip_exps = len(lines)
    scores = [x['score'] for x in map(json.loads, lines)]
else:
    num_skip_exps = 0
    scores = []

with open(OUTPUT_PATH, 'a' if args.append else 'w') as f:
    pbar = tqdm.tqdm(examples[num_skip_exps:], initial=num_skip_exps, total=len(examples))
    for x in pbar:
        question = x['input']
        result = copy.copy(x)
        
        try:
            ans = itf.run(MATH_PROMPT.format(
                question=question), majority_at=args.majority_at, 
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens)
            ans = float(ans)
            score = 1 if abs(ans - x['target']) < 1e-3 else 0
        except Exception as e:
            print(e)
            ans = ''
            score = 0
        scores.append(score)
        
        result['answer'] = ans
        result['score'] = score
        result['generation'] = itf.history
        f.write(json.dumps(result) + '\n')
        
        itf.clear_history()
        f.flush()

print(f'Accuracy - {sum(scores) / len(scores)}')
