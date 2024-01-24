# Our files
import data_preprocessing
import svm 
import visualization
from svm import SVM
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import bert
import DistilBert
import articles
from articles import getArticle
from articles import get_articles
from bert import BERTClassifier
import RF
from RF import RF_model
from RF import RF_evaluate 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKER = 'NVDA'
START = '2023-01-01'
END = '2023-06-01'

DataClass = data_preprocessing.DataHandler()
DataClass.update_remove_neutral()
df = DataClass.get_df()
DataClass.update_remove_stop_words()
df = DataClass.get_df()
visualization.word_count_distribution(df)


df_train = DataClass.get_df_train()
df_val = DataClass.get_df_val()
df_test = DataClass.get_df_val()



# Load the tokenizer and model
# The 2 different models and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model_path = "./results/models/DistilBert_classifier.pth"
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
model = DistilBert.BERTClassifier()

# Load the trained model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()



# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        return "positive" if preds.item() == 1 else "negative"


def predict_sentiment_DistilBertLSTM(text, model, tokenizer, device, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask) # We get logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]  
        return "positive" if prediction == 1 else "negative"



# Classify and label the articles - save it as .CSV
def classify_articles(df_articles, TICKER):
    df_articles['Label'] = df_articles['Headline'].apply(lambda x: predict_sentiment_DistilBertLSTM(x, model, tokenizer, device))
    df_articles.to_csv(f'{TICKER}_classified_articles.csv', index=False)
    print(df_articles)



######################################### 
# classify_articles(df_articles, TICKER)#
#########################################

##########################
# RF_model(DataClass)    #
# RF_evaluate(DataClass) #
##########################

