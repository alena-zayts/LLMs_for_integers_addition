from transformers import AutoTokenizer, AutoModelForCausalLM
from solution_1.prompt_generation import CODE_START_MARKER, EXAMPLES_AMOUNT, QUESTION_START_MARKER, \
    generate_prompt, count_code_answer_length, FUNCTION_CALL
from solution_1.runtime import GenericRuntime
from typing import List
from solutions_evaluation.abstract_solver import AbstractSolver


class Solver1(AbstractSolver):
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
    a = 10097612345678909876542234567890987654321234567890987654321234567890987654323456789
    b = 123456789876543234567898765434567890987654345678998765434567890987654345678987654345678765434567
    expected = a + b
    answer_int, meta_info = solver.calc_sum(a, b)

    print(meta_info)
    print(answer_int)
    print(expected)
