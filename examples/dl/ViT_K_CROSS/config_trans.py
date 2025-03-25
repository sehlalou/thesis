EMB_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 6
PATCH_SIZE = 64
DROPOUT_RATE = 0.45
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