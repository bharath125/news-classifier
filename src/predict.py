
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from config import Config
import argparse
import json

def load_model():
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_SAVE_PATH)
    model     = BertForSequenceClassification.from_pretrained(
        Config.MODEL_SAVE_PATH
    ).to(Config.DEVICE)
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer):
    encoding = tokenizer(
        text,
        max_length     = Config.MAX_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt"
    )
    input_ids      = encoding["input_ids"].to(Config.DEVICE)
    attention_mask = encoding["attention_mask"].to(Config.DEVICE)

    with torch.no_grad():
        outputs     = model(input_ids=input_ids,
                           attention_mask=attention_mask)
        probs       = torch.softmax(outputs.logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    categories = [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
        "comp.windows.x", "misc.forsale", "rec.autos",
        "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
        "sci.crypt", "sci.electronics", "sci.med", "sci.space",
        "soc.religion.christian", "talk.politics.guns",
        "talk.politics.mideast", "talk.politics.misc",
        "talk.religion.misc"
    ]

    category   = categories[predicted.item()]
    confidence = confidence.item() * 100

    # low confidence warning
    needs_review = confidence < 50

    return {
        "category":     category,
        "confidence":   f"{confidence:.1f}%",
        "needs_review": needs_review,
        "message":      "Low confidence — consider human review"
                        if needs_review else "High confidence prediction"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model()
    result = predict(args.text, model, tokenizer)
    print(json.dumps(result, indent=2))
