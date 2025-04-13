DETECTION = False
LOOK_A_HEAD = 180000


WINDOW_SIZE = 4096
EPOCH = 20 # test with 50 for transformer based models
PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_PROC_WORKERS = 12
RANDOM_SEED = 42
DATASET_PATH = "/mnt/iridia/sehlalou/thesis/data/datasets"
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/ViT/saved_models"

# Optimized config
EMB_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6
PATCH_SIZE = 64
DROPOUT_RATE = 0.45
MLP_DIM = 128

# STUDY ON VIT
#EMB_DIM = 16
#NUM_HEADS = 2
#NUM_LAYERS = 2
#PATCH_SIZE = 32
#DROPOUT_RATE = 0.1
#MLP_DIM = 128

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
        "NUM_PROC_WORKERS": NUM_PROC_WORKERS,
        "EMB_DIM": EMB_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "PATCH_SIZE": PATCH_SIZE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "MLP_DIM": MLP_DIM
    }