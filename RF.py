import scipy.stats
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = './results/models/'
REPORT_PATH = './results/models/'

# Random Forest
def RF_model(DataClass):

    model_name = 'RF'
    start_time = time.time()

    # Extract data 
    df_train = DataClass.get_df_train()
    df_val = DataClass.get_df_val()

    X_train = df_train['Text']
    X_val = df_val['Text'] 
    y_train = df_train['Sentiment']
    y_val = df_val['Sentiment']

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Initialize the RF model
    rf = RandomForestClassifier()

    # Define the parameter distribution for randomized search
    param_distributions = {
        'n_estimators': scipy.stats.randint(10, 500),
        'max_depth': scipy.stats.randint(3, 50),
        'min_samples_split': scipy.stats.randint(2, 50)
    }

    # Initialize the randomized search
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=30, cv=5, verbose=10, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Predict and evaluate on the validation set using the best estimator
    best_model = random_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    # Save STATS
    report = classification_report(y_val, y_val_pred)
    end_time = time.time()
    training_duration = end_time - start_time
    report = classification_report(y_val, y_val_pred)
    report = f"Model Training Time: {training_duration:.2f} seconds\n\n" + report

    # Append best parameters to the report
    best_params = random_search.best_params_
    report += "\nBest Parameters:\n"
    for param, value in best_params.items():
        report += f"{param}: {value}\n"

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:")
    print(report)

    # Save the best estimator and report
    dump(best_model, f"{MODEL_PATH}/{model_name}.joblib")
    dump(vectorizer, f'{MODEL_PATH}/{model_name}_trained_vectorizer.joblib')

    with open(f"{REPORT_PATH}/{model_name}_report.txt", "w") as f:
        f.write(report)

    return y_val_pred


# Evaluate the test data for RF Model
def RF_evaluate(DataClass):

    model = joblib.load(MODEL_PATH + 'RF.joblib')
    vectorizer = joblib.load(MODEL_PATH + 'RF_trained_vectorizer.joblib') 
    df_test = DataClass.get_df_test()
    
    test_data = df_test['Text']
    test_labels = df_test['Sentiment']

    # Text Vectorization
    test_data = vectorizer.transform(test_data)
    y_pred = model.predict(test_data)

    print(classification_report(test_labels, y_pred))


