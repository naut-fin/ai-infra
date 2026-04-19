from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score

imdb_data = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenize_dataset = imdb_data.map(tokenize_function, batched=True)
tokenize_dataset = tokenize_dataset.remove_columns(["text"])
tokenize_dataset.set_format("torch")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#--- you can comment this section to run the model with early stopping


train_dataloader = DataLoader(
    tokenize_dataset["train"],
    batch_size = 8,
    shuffle= True # need to add shuffle because we do not want model to learn anything with respect to data order,
    # if data is organized such that first 50 entries in data are positive and last 50 entries are negative in a 100 entry dataset,
    # without shuffle model will learn patterns which does not have any relevance to the task
)

test_dataloader = DataLoader(
    tokenize_dataset["test"],
    batch_size = 8,
    shuffle = False
)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

model.to(device)

num_epochs = 1 # define the number of times the full dataset would forward pass through the model

optimizer = AdamW(model.parameters(), lr=5e-5)


# using manual training loop

for epoch in range(num_epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
        batch = {k:v.to(device) for k, v in batch.items()}
        outputs= model(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels = batch["label"]
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if step%100==0:
            print(f"Epoch:{epoch}, Step:{step}, Loss:{loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"]
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions==batch["label"]).sum().item()
            total += batch["label"].size(0)

    accuracy_score = correct/total
    print(f"Epoch:{epoch} , Accuracy Score:{accuracy_score:.4f}")

#--------------------------------------------------------- comment till here

model_b = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model_b.to(device)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {"accuracy":accuracy_score(labels, preds)}


training_args = TrainingArguments(
    output_dir="./results_with_weight_decay_early_stopping",
    per_device_train_batch_size=64, # tweak this to reduce OOM issues batch size was able to run this on an L4 GPU with 22.5 GB RAM utilization was 12.7 GB
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)


trainer = Trainer(
    model=model_b,
    args=training_args,
    train_dataset=tokenize_dataset["train"],
    eval_dataset=tokenize_dataset["test"],
    compute_metrics=compute_metrics,
    callback=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()

trainer.save_model("./imdb_sentiment_model") # step to save the model
tokenizer.save_pretrained("./imdb_tokenizer") # step to save the tokenizer









