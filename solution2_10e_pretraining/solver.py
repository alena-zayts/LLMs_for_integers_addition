from typing import Tuple
import torch
import glob
import os
from training import T5Finetuner
from utils import translate_task, convert_from_10ebased


class Solver2:
    def __init__(self):
        # checkpoint_path = 'pretrained_model.ckpt'
        cur_dir = os.path.basename(os.getcwd())


        if cur_dir == 'solution2_10e_pretraining\\':
            root_dir = '.'
        else:
            root_dir = 'solution2_10e_pretraining\\'

        checkpoint_path = glob.glob(f"{root_dir}*.ckpt")[0]

        self.model = T5Finetuner.load_from_checkpoint(checkpoint_path,
                                                      train_dataloader=None,
                                                      val_dataloader=None,
                                                      test_dataloader=None,
                                                      map_location=torch.device('cpu')  # change it if you have >= 1 gpu
                                                      )

    def calc_sum(self, a: int, b: int) -> Tuple[int, dict]:
        translated_task = translate_task(a, b)
        model_answer = self.model.predict(translated_task['question'])
        converted_model_answer = int(convert_from_10ebased(model_answer))

        translated_task['real_model_answer'] = model_answer
        translated_task['converted_model_answer'] = converted_model_answer

        return converted_model_answer, translated_task


if __name__ == '__main__':
    solver = Solver2()
    a = 20765167898761789
    b = 3132678982783293
    expected = a + b
    answer_int, meta_info = solver.calc_sum(a, b)

    print(f'Meta-info: {meta_info}')
    print(f"Model's answer: {answer_int}")
    print(f'Expected answer: {expected}')
    print(f'Correct?: {expected == answer_int}')
