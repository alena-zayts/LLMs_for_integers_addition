from typing import Tuple
import torch
import glob
import os
from training import T5Finetuner
from utils import translate_task, convert_from_10ebased


class Solver2:
    def __init__(self):

        checkpoints = glob.glob("*.ckpt", recursive=True)
        if not checkpoints:
            checkpoints = glob.glob("solution2_10e_pretraining\\*.ckpt", recursive=True)
            if not checkpoints:
                raise ValueError('Downlad and save a pretrained model as solution2_10e_pretraining/pretrained_model.ckpt first! (https://drive.google.com/file/d/1kcatj1OOMO8AU1DP6UW4kj82OxikBb4a/view?usp=share_link)')

        checkpoint_path = checkpoints[0]

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
