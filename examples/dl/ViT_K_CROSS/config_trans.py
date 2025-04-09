EMB_DIM = 16
NUM_HEADS = 2
NUM_LAYERS = 2
PATCH_SIZE = 32
DROPOUT_RATE = 0.1
MLP_DIM = 128
DATASET_PATH = "data/datasets/"
LOG_DL_PATH = "/mnt/iridia/sehlalou/thesis/examples/dl/ViT_K_CROSS/saved_models"


def get_dict():
    return {
        "EMB_DIM": EMB_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "PATCH_SIZE": PATCH_SIZE,
        "DROPOUT_RATE": DROPOUT_RATE,
        "MLP_DIM": MLP_DIM,
    }