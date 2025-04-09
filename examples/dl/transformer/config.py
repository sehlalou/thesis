WINDOW_SIZE = 640
EMB_DIM = 16
NUM_LAYERS = 2
NUM_HEADS = 2
MLP_DIM = 128
DROPOUT_RATE = 0.1
RANDOM_SEED = 42
EPOCH = 20
PATIENCE = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_PROC_WORKERS_DATA = 12
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/transformer/saved_models"
DATASET_PATH = "/mnt/iridia/sehlalou/thesis/data/datasets"


def get_dict():
    return {
        "WINDOW_SIZE": WINDOW_SIZE,
        "EMB_DIM": EMB_DIM,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "MLP_DIM": MLP_DIM,
        "DROPOUT_RATE":DROPOUT_RATE,
        "RANDOM_SEED": RANDOM_SEED,
        "EPOCH": EPOCH,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS_DATA
    }