# News Article Cla

Classifies news articles into 20 categories using fine-tuned BERT.

## Results
- 20 categories (20 Newsgroups dataset)
- 64.4% accuracy on test set
- Trained on 4000 balanced examples (200 per class)

## Model
Hosted on HuggingFace Hub:
https://huggingface.co/YOUR_HF_USERNAME/news-classifier

Load directly:
```python
from transformers import BertForSequenceClassification, BertTokenizer
model     = BertForSequenceClassification.from_pretrained("YOUR_HF_USERNAME/news-classifier")
tokenizer = BertTokenizer.from_pretrained("YOUR_HF_USERNAME/news-classifier")
```

## Setup
```bash
pip install -r requirements.txt
```

## Run Training
```bash
cd src
python train.py
```

## Run Prediction
```bash
cd src
python predict.py --text "NASA launches rocket to Mars"
```

## Categories
alt.atheism, comp.graphics, comp.os.ms-windows.misc,
comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x,
misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball,
rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space,
soc.religion.christian, talk.politics.guns, talk.politics.mideast,
talk.politics.misc, talk.religion.misc
