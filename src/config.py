
import torch

class Config:
    MODEL_NAME        = "bert-base-uncased"
    NUM_CLASSES       = 20
    MAX_LEN           = 256
    BATCH_SIZE        = 16
    EPOCHS            = 6
    LEARNING_RATE     = 1e-5
    WEIGHT_DECAY      = 0.01
    WARMUP_RATIO      = 0.1
    EXAMPLES_PER_CLASS = 200
    TEST_SIZE         = 2000
    MODEL_SAVE_PATH   = "models/news_classifier"
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
