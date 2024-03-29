from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_generation import QUESTION_START_MARKER, generate_prompt, \
    count_code_answer_length, FUNCTION_CALL
from runtime import GenericRuntime
from typing import List


class Solver1:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        self.prompt = generate_prompt()
        self.runtime = GenericRuntime()

    def execute(self, code: List[str]):
        self.runtime.exec_code('\n'.join(code))
        return self.runtime.eval_code(FUNCTION_CALL)

    def calc_sum(self, a: int, b: int) -> (int, dict):
        question = self.prompt.format(a=a, b=b)
        expected_code_answer_length = count_code_answer_length(a, b)

        input_ids = self.tokenizer(question, return_tensors="pt").input_ids
        generated_ids = self.model.generate(input_ids, max_length=len(question) + expected_code_answer_length)

        full_model_answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        real_model_answer = full_model_answer[len(question):]
        code_model_answer = real_model_answer.split(QUESTION_START_MARKER)[0]

        extra_tab_len = 4
        code_model_answer = code_model_answer.split('\n')
        for i in range(len(code_model_answer)):
            if len(code_model_answer[i]) > extra_tab_len:
                code_model_answer[i] = code_model_answer[i][extra_tab_len:]

        # do not change order
        answer_int = self.execute(code_model_answer)
        code_model_answer = '\n'.join(code_model_answer)

        meta_info = {
            'question': question,
            'full_model_answer': full_model_answer,
            'code_model_answer': code_model_answer,
            'answer_int': answer_int
        }

        return answer_int, meta_info


if __name__ == '__main__':
    solver = Solver1()
    a = 2
    b = 3
    expected = a + b
    answer_int, meta_info = solver.calc_sum(a, b)

    print(f'Meta-info: {meta_info}')
    print(f"Model's answer: {answer_int}")
    print(f'Expected answer: {expected}')
    print(f'Correct?: {expected == answer_int}')
