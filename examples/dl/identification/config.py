WINDOW_SIZE = 2048 
EMB_DIM = 16
NUM_HEADS = 2
NUM_LAYERS = 2
PATCH_SIZE = 32
DROPOUT_RATE = 0.1
MLP_DIM = 128
EPOCH = 2 
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_PROC_WORKERS = 12
RANDOM_SEED = 42
DATASET_PATH = "/mnt/iridia/sehlalou/thesis/data-v2/dataset.csv"
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/identification/saved_models"



def get_dict():
    return {
        "WINDOW_SIZE": WINDOW_SIZE,
        "EMB_DIM": EMB_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "PATCH_SIZE": PATCH_SIZE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "MLP_DIM": MLP_DIM,
        "EPOCH": EPOCH,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS,
        "RANDOM_SEED": RANDOM_SEED,
        "DATASET_PATH": DATASET_PATH,
        "LOG_DL_PATH": LOG_DL_PATH
    }