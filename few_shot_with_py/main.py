import openai
from baсkend import MyProgramInterface
from prompts_formatting import generate_prompt
from api_key import API_KEY
import argparse

openai.api_key = API_KEY

SUM_PROMPT = generate_prompt()


# models: 'text-davinci-003', 'gpt-3.5-turbo'
class Settings:
    def __init__(self, model_name='text-davinci-003', test_examples_path=f'test_examples.jsonl',
                 results_path='test_results.jsonl',
                 continue_experiment=False, temperature=0.0, top_p=1.0, max_tokens=256, majority_at=None):
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

        # use voting or not
        self.majority_at = majority_at


settings = Settings()

interface = MyProgramInterface(stop='\n\n\n', get_answer_expr='solution()', model=settings.model_name, verbose=True)


def summ(a: int, b: int) -> int:
    question = SUM_PROMPT.format(a=a, b=b)

    try:
        ans = interface.run(
            question,
            majority_at=settings.majority_at,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        ans = int(ans)

    except Exception as e:
        print(e)
        ans = ''

    interface.clear_history()

    generation = interface.history
    print(ans)
    print(generation)

    return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    parser.add_argument('--a', type=str, required=True, help='first integer summand')
    parser.add_argument('--b', type=str, required=True, help='second integer summand')

    args = parser.parse_args()

    try:
        res = summ(int(args.a), int(args.b))
        print(f'{args.a} + {args.b} = {res}')

    except Exception as e:
        print(e)
