WINDOW_SIZE = 640
TRAINING_STEP = WINDOW_SIZE // 2
TESTING_STEP = WINDOW_SIZE
RANDOM_SEED = 42
EPOCH = 20 # test with 50 for transformer based models
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_PROC_WORKERS_DATA = 12

# Forecast (Lookahead)
#PRE_AF_WINDOW = 4320000 # 6 hours
#PRE_AF_WINDOW = 2880000 # 4 hours  
#PRE_AF_WINDOW = 1440000 # 2 hours
#PRE_AF_WINDOW = 720000 # 1 hour
#PRE_AF_WINDOW = 360000 # 30 min
#PRE_AF_WINDOW = 180000 # 15 min
PRE_AF_WINDOW = 60000 # 5 min


DATASET_PATH = "/mnt/iridia/sehlalou/thesis/data/datasets"
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models"

def get_dict():
    return {
        "WINDOW_SIZE": WINDOW_SIZE,
        "TRAINING_STEP": TRAINING_STEP,
        "TESTING_STEP": TESTING_STEP,
        "RANDOM_SEED": RANDOM_SEED,
        "EPOCH": EPOCH,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS_DATA
    }