import argparse
import glob
import json
import os
import pytorch_lightning as pl
import random
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import List

from training_settings import *
from utils import *

class T5Finetuner(pl.LightningModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader):
        super(T5Finetuner, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def prepare_batch(self, questions: List[str], answers: List[str]) -> List[str]:

        input_dict = self.tokenizer.batch_encode_plus(
            list(questions), padding=True, truncation=False, return_tensors='pt')

        labels = self.tokenizer.batch_encode_plus(
            list(answers), padding=True, truncation=False, return_tensors='pt')['input_ids']

        assert input_dict['input_ids'].shape[1] < MAX_SEQ_LEN
        assert labels.shape[1] < MAX_SEQ_LEN

        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        labels = labels.to(self.model.device)

        return input_ids, attention_mask, labels

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_nb):
        questions, correct_answers = batch

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print(questions[0])
            print(correct_answers[0])

        input_ids, attention_mask, labels = self.prepare_batch(questions=questions, answers=correct_answers)

        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[0]

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def predict(self, question):
        input_ids, attention_mask, _ = self.prepare_batch(questions=[question], answers=[question])
        batch_outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                                            max_length=MAX_SEQ_LEN)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        return predicted_answers[0]

    def inference_step(self, batch, batch_nb: int):
        questions, correct_answers = batch

        input_ids, attention_mask, _ = self.prepare_batch(questions=questions, answers=correct_answers)
        batch_outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                                            max_length=MAX_SEQ_LEN)

        predicted_answers = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in batch_outputs]

        exact_matches = [
            compute_exact_match(predicted_answer=predicted_answer, correct_answer=correct_answer)
            for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)]

        # Log every power of two.
        if batch_nb & (batch_nb - 1) == 0:
            print('\nQuestion:', questions[0])
            print('Correct:  ', correct_answers[0])
            print('Predicted:', predicted_answers[0].encode('utf-8'))
            print('Exact?', exact_matches[0])

        metrics = {'exact_matches': exact_matches}
        return metrics

    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def validation_epoch_end(self, outputs):
        print('QQ: in validation_epoch_end')
        exact_matches = []
        for x in outputs:
            exact_matches.extend(x['exact_matches'])
        exact_match = sum(exact_matches) / len(exact_matches)

        metrics = {'val_exact_match': exact_match}

        output = metrics.copy()
        output['progress_bar'] = metrics

        # added
        self.log('val_exact_match', exact_match, prog_bar=True)

        return output

    def test_epoch_end(self, outputs):
        exact_matches = []
        for x in outputs:
            exact_matches.extend(x['exact_matches'])
        exact_match = sum(exact_matches) / len(exact_matches)

        metrics = {'test_exact_match': exact_match}
        print('test_exact_match', exact_match)

        output = metrics.copy()
        output['progress_bar'] = metrics
        self.log('test_exact_match', exact_match, prog_bar=True)

        return output

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def get_optimizer(self):
        optimizer = getattr(torch.optim, OPTIMIZER_NAME)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = optimizer(optimizer_grouped_parameters, lr=LR, weight_decay=WEIGHT_DECAY)

        print(f'=> Using {OPTIMIZER_NAME} optimizer')

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE,
                                                    gamma=GAMMA)
        print(f'=> Using StepLR (step_size = {STEP_SIZE}, gamma = {GAMMA}) scheduler')

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer


class MyDataset(Dataset):
    def __init__(self, n_examples: int, min_digits: int, max_digits: int):

        self.max_digits = max_digits

        # if balance:
        self.examples = []
        for _ in range(n_examples):
            example = []
            for _ in range(2):
                max_digits_i = random.randint(min_digits, max_digits)
                min_number = int((max_digits_i - 1) * '9') + 1
                max_number = int(max_digits_i * '9')
                example.append(random.randint(min_number, max_number))
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        first_term, second_term = self.examples[idx]
        translated_task = translate_task(first_term, second_term)

        return translated_task['question'], translated_task['expected_result_str']


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    pl.seed_everything(SEED)

    dataset_train = MyDataset(n_examples=TRAIN_SIZE, min_digits=MIN_DIGITS_TRAIN, max_digits=MAX_DIGITS_TRAIN)
    dataset_val = MyDataset(n_examples=VAL_SIZE, min_digits=MIN_DIGITS_TRAIN, max_digits=MAX_DIGITS_TRAIN)
    dataset_test = MyDataset(n_examples=TEST_SIZE, min_digits=MIN_DIGITS_TEST, max_digits=MAX_DIGITS_TEST)

    train_dataloader = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=VAL_BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(dataset_test, batch_size=VAL_BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR, filename='{epoch}-{val_exact_match:.4f}',
        verbose=False, save_last=True, save_top_k=1, mode='max', monitor='val_exact_match',
        save_weights_only=False, every_n_epochs=CHECK_VAL_EVERY_N_EPOCH,
        # save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(
        precision=32,
        callbacks=[checkpoint_callback],
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        accumulate_grad_batches=32,
        gradient_clip_val=1.0,
        amp_level='O0',
        amp_backend='apex',
        gpus=GPUS)

    model = T5Finetuner(train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        test_dataloader=test_dataloader)

    trainer.fit(model)



    # checkpoint_path = glob.glob(os.path.join(OUTPUT_DIR, '*.ckpt'))[0]
    checkpoint_path = '/content/first_results/epoch=11-val_exact_match=1.0000.ckpt'
    model = T5Finetuner.load_from_checkpoint(checkpoint_path,
                                             train_dataloader=train_dataloader,
                                             val_dataloader=val_dataloader,
                                             test_dataloader=test_dataloader)

    results = trainer.test(model)

    output = {'seed': SEED,
              'max_digits_train': MAX_DIGITS_TRAIN,
              'max_digits_test': MAX_DIGITS_TEST,
              'test_exact_match_': results[0]}

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as fout:
        json.dump(output, fout)

    print('Done!')


if __name__ == '__main__':
    train()

