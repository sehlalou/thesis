DETECTION = False
#LOOK_A_HEAD = 2880000
LOOK_A_HEAD = 3600
WINDOW_SIZE = 2048
EPOCH = 20 # test with 50 for transformer based models
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_PROC_WORKERS = 12
RANDOM_SEED = 42
DATASET_PATH = "/mnt/iridia/sehlalou/thesis/data/datasets"

def get_dict():
    return {
        "DETECTION": DETECTION,
        "LOOK_A_HEAD": LOOK_A_HEAD,
        "WINDOW_SIZE": WINDOW_SIZE,
        "RANDOM_SEED": RANDOM_SEED,
        "EPOCH": EPOCH,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS
    }