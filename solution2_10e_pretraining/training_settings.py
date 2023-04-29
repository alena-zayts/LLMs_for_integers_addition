SEED = 1

NUM_WORKERS = 4
LR = 3e-4
WEIGHT_DECAY = 5e-5
GAMMA = 1.0  # 0.1
STEP_SIZE = 1000

OPTIMIZER_NAME = 'AdamW'
MAX_SEQ_LEN = 512
CHECK_VAL_EVERY_N_EPOCH = 2
MAX_EPOCHS = 20
GPUS = 1


OUTPUT_DIR = 'first_results'
MODEL_NAME = 't5-base'  # t5-small, t5-base
MIN_DIGITS_TRAIN = 2
MAX_DIGITS_TRAIN = 15
MIN_DIGITS_TEST = 2
MAX_DIGITS_TEST = 15

TRAIN_SIZE = 100000
TRAIN_BATCH_SIZE = 4
VAL_SIZE = 10000
VAL_BATCH_SIZE = 32
TEST_SIZE = 10000

