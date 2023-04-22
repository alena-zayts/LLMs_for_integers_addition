from typing import *
import copy
import openai
import time
import io
from contextlib import redirect_stdout


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars['answer']



class MyProgramInterface:

    def __init__(
            self,
            model: str = 'code-davinci-002',
            stop: str = '\n\n\n',
            get_answer_expr: str = None,
            verbose: bool = False
    ) -> None:

        self.model = model
        self.stop = stop
        self.answer_expr = get_answer_expr
        self.verbose = verbose

        self.runtime = GenericRuntime()
        self.history = []

    def clear_history(self):
        self.history = []

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    def generate(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 512, majority_at: int = None, ):
        gens = call_gpt(prompt, model=self.model, stop=self.stop,
                        temperature=temperature, top_p=top_p, max_tokens=max_tokens, majority_at=majority_at, )
        if self.verbose:
            print(gens)
        code = self.process_generation_to_code(gens)
        self.history.append(gens)
        return code

    def execute(self, code: Optional[List[str]] = None):
        code = code if code else self.code
        self.runtime.exec_code('\n'.join(code))
        return self.runtime.eval_code(self.answer_expr)


    def run(self, prompt: str, time_out: float = 10, temperature: float = 0.0, top_p: float = 1.0,
            max_tokens: int = 512, majority_at: int = None):
        code_snippets = self.generate(prompt, majority_at=majority_at, temperature=temperature, top_p=top_p,
                                      max_tokens=max_tokens)

        results = []
        for code in code_snippets:
            with timeout(time_out):
                try:
                    exec_result = self.execute(code)
                except Exception as e:
                    print(e)
                    continue
                results.append(exec_result)
        counter = Counter(results)
        return counter.most_common(1)[0][0]


# GPT-3 API
def call_gpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
             max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            if model.startswith('gpt-4') or model.startswith('gpt-3.5-turbo'):
                ans = chat_api(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions,
                    best_of=requested_completions)
            else:
                ans = completions_api(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions,
                    best_of=requested_completions)
            completions.extend(ans)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            time.sleep(min(i ** 2, 60))
    raise RuntimeError('Failed to call GPT API')

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        pass
        # TODO: изменила тк на винде нет
        # signal.signal(signal.SIGALRM, self.timeout_handler)
        # signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        pass
        #signal.alarm(0)


def chat_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.ChatCompletion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
            {'role': 'user', 'content': prompt}],
        temperature=temperature,
        top_p=top_p,
        n=n)
    return [choice['message']['content'] for choice in ans['choices']]


def completions_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        n=n,
        best_of=best_of)
    return [choice['text'] for choice in ans['choices']]
