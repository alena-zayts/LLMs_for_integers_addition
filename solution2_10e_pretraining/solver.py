from typing import Tuple
from training import T5Finetuner
from utils import translate_task

class Solver2():
    def __init__(self):
        checkpoint_path = '/content/first_results/epoch=11-val_exact_match=1.0000.ckpt'

        self.model = T5Finetuner.load_from_checkpoint(checkpoint_path,
                                                      train_dataloader=None,
                                                      val_dataloader=None,
                                                      test_dataloader=None)

    def calc_sum(self, a: int, b: int) -> Tuple[int, dict]:
        translated_task = translate_task(a, b)
        model_answer = self.model.predict(translated_task['question'])

        return model_answer, translated_task

        # real_model_answer = full_model_answer[len(question):]
        # code_model_answer = real_model_answer.split(QUESTION_START_MARKER)[0]

        # extra_tab_len = 4
        # code_model_answer = code_model_answer.split('\n')
        # for i in range(len(code_model_answer)):
        #     if len(code_model_answer[i]) > extra_tab_len:
        #         code_model_answer[i] = code_model_answer[i][extra_tab_len:]

        # # do not change order
        # answer_int = self.execute(code_model_answer)
        # code_model_answer = '\n'.join(code_model_answer)

        # meta_info = {
        #     'question': question,
        #     'full_model_answer': full_model_answer,
        #     'code_model_answer': code_model_answer,
        #     'answer_int': answer_int
        # }

        # return answer_int, meta_info


if __name__ == '__main__':
    solver = Solver2()
    a = 2
    b = 3
    answer_int, meta_info = solver.calc_sum(a, b)

    print(meta_info)
    print(answer_int)
