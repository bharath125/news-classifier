from transformers import BertForSequenceClassification
from config import Config

def create_model():
    """
    Creates BERT model for sequence classification.
    Returns model moved to correct device.
    """
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels = Config.NUM_CLASSES
    ).to(Config.DEVICE)
    return model

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
