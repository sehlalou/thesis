EMB_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
PATCH_SIZE = 32
DROPOUT_RATE = 0.1
MLP_DIM = 512
DATASET_PATH = "data/datasets/"
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/training_transformer"


def get_dict():
    return {
        "EMB_DIM": EMB_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "PATCH_SIZE": PATCH_SIZE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "MLP_DIM": MLP_DIM,
    }