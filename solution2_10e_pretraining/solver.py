from typing import Tuple
from training import T5Finetuner
from utils import translate_task, convert_from_10ebased
from solutions_evaluation.abstract_solver import AbstractSolver


class Solver2(AbstractSolver):
    def __init__(self):
        checkpoint_path = '/content/first_results/epoch=11-val_exact_match=1.0000.ckpt'

        self.model = T5Finetuner.load_from_checkpoint(checkpoint_path,
                                                      train_dataloader=None,
                                                      val_dataloader=None,
                                                      test_dataloader=None)

    def calc_sum(self, a: int, b: int) -> Tuple[int, dict]:
        translated_task = translate_task(a, b)
        model_answer = self.model.predict(translated_task['question'])
        converted_model_answer = int(convert_from_10ebased(model_answer))

        translated_task['real_model_answer'] = model_answer
        translated_task['converted_model_answer'] = converted_model_answer

        return converted_model_answer, translated_task


if __name__ == '__main__':
    solver = Solver2()
    a = 2
    b = 3
    answer_int, meta_info = solver.calc_sum(a, b)

    print(meta_info)
    print(answer_int)
