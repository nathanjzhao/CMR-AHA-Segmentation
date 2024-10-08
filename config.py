from pathlib import Path

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "validation score"},
    "parameters": {
        "lr": {
            "min": 1e-6,
            "max": 1e-2,
        },
        "scheduler_patience": {
            "min": 5,
            "max": 100,
        },
        "weight_decay": {
            "min": 1e-6,
            "max": 1e-3,
        },
        "momentum": {
            "min": 0.9,
            "max": 0.999,
        },
        # HYPERPARAMETERS
        "sigma": {
            "min": 5,
            "max": 7,
        },
        "rotation": {
            "min": 1,
            "max": 30,
        },
        "translation": {
            "min": 0.05,
            "max": 0.2,
        },
        "scale": {
            "min": 1.0,
            "max": 2.5,
        },
        "contrast": {
            "min": 1.0,
            "max": 3.0,
        },
        "CE": {
            "min": 0.7,
            "max": 1.0,
        },
        "tversky_beta": {
            "min": 0.2,
            "max": 0.95,
        },
    },
}

# pre- 6/30
# sweep_config = {
#     "method": "bayes",
#     "metric": {"goal": "maximize", "name": "validation score"},
#     "parameters": {
#         "lr" : {
#             "min" : 1e-6,
#             "max" : 1e-2,
#         },
#         "scheduler_patience": {
#             "min" : 5,
#             "max" : 100,
#         },
#         "weight_decay": {
#             "min" : 1e-12,
#             "max" : 1e-4,
#         },
#         "momentum": {
#             "min" : 0.9,
#             "max" : 0.999,
#         },

#         # HYPERPARAMETERS
#         "sigma": {
#             "min": 2,
#             "max": 7,
#         },
#         "rotation": {
#             "min": 1,
#             "max": 15,
#         },
#         "translation": {
#             "min": .05,
#             "max": .12,
#         },
#         "scale": {
#             "min": 1.0,
#             "max": 2.5,
#         },
#         "contrast": {
#             "min": 1.0,
#             "max": 3.0,
#         },
#         "CE": {
#             "min": 0.3,
#             "max": 0.95,
#         },
#         "tversky_beta": {
#             "min": 0.3,
#             "max": 0.95,
#         },
#     }
# }


# Summer of Segmentation data
data_path = "./data/unswapped_data/train"
val_data_path = "./data/unswapped_data/val"

# data_path = './data/original_standard_labels/train'
# val_data_path = './data/original_standard_labels/val'

checkpoint_path = Path("./checkpoints/")
save_checkpoint = True

random_seed = 42
results_path = "./model_tests"
no_midpoint = True
test_name_prefix = ""
filter_level = 0
record_spread = False

use_MD = False  # Set to False if you don't want to use MD
use_E1 = False  # Set to False if you don't want to use E1
use_E1_xyz = False  # Utilizes ||x|+|y|-|z|| instead of all x y z

batch_size = 8
num_epochs = 100
# num_features = 8
# relu = False
# dropout = 0.5
# early_stopping_patience = 20

scheduler_patience = 20
lr = 1e-6
weight_decay = 1e-8
momentum = 0.999
bilinear = False
sigma = 5

flipping = False
amp = False
