import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import TFTrainingArguments
import tensorflow as tf
import pickle
from torch.utils.data import DataLoader, Dataset
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
import time

from transformers import DistilBertModel, DistilBertTokenizer

# Load data_handler class 
def loadDataHandler(class_path):
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        data_handler.get_class_name = class_path
        print(data_handler.get_class_name)
        return data_handler


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    

# Create model class
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        # Hardcoded parameters
        bert_model_name = "distilbert-base-uncased"
        num_classes = 2
        hidden_dim = 128
        lstm_layers = 1
        bidirectional = True

        # Initialize layers
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        last_hidden_state = lstm_output[:, -1, :]
        logits = self.fc(last_hidden_state)

        return logits


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # Directly use logits

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()



def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # Directly use logits

            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)



def predict_sentiment(text, model, tokenizer, device, max_length=45):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"


if __name__ == "__main__":

    """
    This part, is used the train the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    bert_model_name = 'bert-base-uncased'

    # Hyperparameters
    num_classes = 2
    max_length = 45 
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.0001 
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    DataClass = loadDataHandler('ClassData2')
    

    # Creating the data 
    df_train = DataClass.get_df_train()
    df_train = df_train.reset_index(drop=True)
    df_train_labels = df_train['Sentiment']
    df_train_text = df_train['Text']
    
    df_val = DataClass.get_df_val()
    df_val = df_val.reset_index(drop=True)
    df_val_labels = df_val['Sentiment']
    df_val_text = df_val['Text']
    
    train_dataset = TextClassificationDataset(df_train_text, df_train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(df_val_text, df_val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    
    # Initialize the model with the same parameters used for training
    model = BERTClassifier()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_accuracy = 0.0
    best_model_state = None
    training_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)


        # Save the model with the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()  # Save the best model state


            # Save classifier report for the best model
            with open("best_model_report.txt", "w") as file:
                file.write(f"Epoch {epoch + 1}/{num_epochs}\n")
                file.write(f"Validation Accuracy: {accuracy:.4f}\n")
                file.write("Classification Report:\n")
                file.write(report)
            print(f"New best model found at epoch {epoch + 1}!")
                

    # torch.save(model.state_dict(), "bert_classifier.pth")
    torch.save(best_model_state, "DistilBert_classifier.pth")


    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
