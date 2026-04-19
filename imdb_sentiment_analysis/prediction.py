from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./imdb_tokenizer")

model = AutoModelForSequenceClassification.from_pretrained("./imdb_sentiment_model")

model.to(device)

examples = [
    "This movie was fantastic and deeply moving.",
    "This movie was boring and painfully slow.",
    "I expected it to be awful, but it was actually great.",
    "Great, another masterpiece of terrible acting.",
    "This movie would top the list of best movies from the bottom"
]

inputs = tokenizer(examples, return_tensors="pt", truncation=True, padding=True)

inputs = {k:v.to(device) for k,v in inputs.items()}



with torch.no_grad():
    outputs = model(**inputs)
    predictions=torch.argmax(outputs.logits, dim=-1)

    label_map = {0: "negative", 1: "positive"}

    for text, pred in zip(examples, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {label_map[pred.item()]}")
        print("-" * 50)



# Text: This movie was fantastic and deeply moving.
# Prediction: positive
# --------------------------------------------------
# Text: This movie was boring and painfully slow.
# Prediction: negative
# --------------------------------------------------
# Text: I expected it to be awful, but it was actually great.
# Prediction: positive
# --------------------------------------------------
# Text: Great, another masterpiece of terrible acting.
# Prediction: negative
# --------------------------------------------------
# Text: This movie would top the list of best movies from the bottom
# Prediction: positive
# --------------------------------------------------



