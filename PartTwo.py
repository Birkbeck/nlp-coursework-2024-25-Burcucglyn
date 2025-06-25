from pathlib import Path
import pandas as pd

#vectorization and splitting for B
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split   

#random forest and SVM for C
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE  #for oversampling the minority class
#New approach for C 
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#for custom tokenizer in E - that's the part I tried to use vectorization and tokenization with nltk and spacy but it was not working as expected
#To make the proiocess faster also add disable the parsel but the result was the same as using the custom tokenizer
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

import re
from spacy.lang.en.stop_words import STOP_WORDS

# Custom tokenizer function to remove stop words and tokenize text to do that got help on ChatGBP/CoPilot
#Before I used this also tried with nltk and spacy but it was not working as expected takes too long to process the text and eventually crushes the kernel
def custom_tokenizer(text):
    return [token for token in re.findall(r'\b\w\w+\b', text.lower()) if token not in STOP_WORDS]

from tqdm import tqdm
tqdm.pandas()

# ========== PART A: Data Cleaning ==========
'''A) Data cleaning: read, filter parties, keep only 'Speech', filter by length.'''

print("\n========== PART A: Data Cleaning ==========")

def read_hansard(path=Path.cwd() / "texts" / "hansard40000.csv"):
    """Reads the hansard40000.csv dataset"""
    df = pd.read_csv(path)
    # Change the 'Labour (Co-op)' value in 'party' column to 'Labour'
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    return df

# Check if the function is working
df = read_hansard("/Users/burdzhuchaglayan/Desktop/Msci_DS/summer term/natural language processing/coursework needs to be submitted at 3rd of july/p2-texts/hansard40000.csv")
print(df.head())

def filter_parties(df):
    """Keep only main 4 parties."""
    main_parties = ['Labour', 'Conservative', 'Liberal Democrat', 'Independent']
    df = df[df['party'].isin(main_parties)]
    return df

df = filter_parties(df)
print(df['party'].unique())  # Should only show the main 4 parties
print(df.shape)

def filter_speech(df):
    """Keep only rows where speech_class is 'Speech'."""
    df = df[df['speech_class'] == 'Speech']
    return df

df = filter_speech(df)
print(df['speech_class'].unique())

def filter_speech_length(df):
    """Keep only speeches >= 1000 chars."""
    df = df[df['speech'].str.len() >= 1000]
    return df

df = filter_speech_length(df)
print(df.shape)  # Print the shape of the dataframe to see how many rows are left

def splitdf(X, y, test_size = 0.2, random_state = 26):
    """Splits the dataframe into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ========== PART B: Vectorization & Train/Test Split ==========
'''B) Vectorise speeches, remove stopwords, max_features=3000, stratified split.'''

print("\n========== PART B: Vectorization & Train/Test Split ==========")

def vectorize_speeches(df):
    """Vectorizes the speeches using TfidfVectorizer and splits the data into train and test sets."""
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english", max_features=3000
    )
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26, stratify=y
    )
    feature_names = vectorizer.get_feature_names_out()
    return X_train, X_test, y_train, y_test, feature_names

X_train, X_test, y_train, y_test, feature_names = vectorize_speeches(df)

# Check the shapes of the train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("First 10 feature names:", feature_names[:10])



# ========== PART C:Classifiers (RandomForest & SVM) ==========
'''C) Train RandomForest and SVM, print macro F1 and classification report. '''
print("\n========== PART C:Classifiers (RandomForest & SVM) ==========")

def classifiers_models (X_train, X_test, y_train, y_test):
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=26, class_weight='balanced') #additional add for non biased model added class_weight='balanced'
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_f1_score = f1_score(y_test, rf_predictions, average='macro')
    print("Random Forest Macro F1 Score:", rf_f1_score)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions, zero_division=0)) 

    # Linear SVM
    svm_model = SVC(kernel='linear', random_state=26, class_weight='balanced') #additional add for non biased model added class_weight='balanced'
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_f1_score = f1_score(y_test, svm_predictions, average='macro')
    print("SVM Macro F1 Score:", svm_f1_score)
    print("SVM Classification Report:\n", classification_report(y_test, svm_predictions, zero_division=0))

# check
classifiers_models(X_train, X_test, y_train, y_test)

# # Apply SMOTE to the training data only for non biased model
# smote = SMOTE(random_state=26)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print("After SMOTE oversampling:")
# print("y_train_resampled value counts:\n", pd.Series(y_train_resampled).value_counts())

# # Now train your models on the resampled data
# classifiers_models(X_train_resampled, X_test, y_train_resampled, y_test)


# ========== PART D,E: Scikit- Pipelines ==========
'''Improved pipeline with SMOTE for later parts (D/E) --- '''
print("\n========== PART D,E: Improved pipeline with SMOTE  ==========")

def pipe_train_model(X_train, X_test, y_train, y_test, ngram_range=(1,1), tokenizer=None):
    """Trains RandomForest and SVM pipelines with given ngram_range and optional custom tokenizer, prints classification reports."""  
    # Random Forest Pipeline
    rf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram_range, # add this line to include n-grams and re-use the code part D
            tokenizer=tokenizer,     # add this line to include custom tokenizer and re-use the code part E
            sublinear_tf=True, max_df=0.5, min_df=5, 
            stop_words="english" if tokenizer is None else None, # if tokenizer is not None, no need to remove stop words part E
            max_features=3000)),
        ('smote', SMOTE(random_state=26)), # added smote for oversampling
        ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=26))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_preds = rf_pipe.predict(X_test)
    print("Random Forest Pipeline Macro F1:", f1_score(y_test, rf_preds, average='macro'))
    # set zero_division=0 to avoid UndefinedMetricWarning
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds, zero_division=0))


    # SVM Pipeline with SMOTE
    svm_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram_range, # add this line to include n-grams and re-use the code part D
            tokenizer=tokenizer,     # add this line to include custom tokenizer and re-use the code part E
            sublinear_tf=True, max_df=0.5, min_df=5, 
            stop_words="english" if tokenizer is None else None, # if tokenizer is not None, no need to remove stop words part E
            max_features=3000)),
        ('smote', SMOTE(random_state=26)), # added smote for oversampling
        ('clf', SVC(kernel='linear', class_weight='balanced', random_state=26))
    ])
    svm_pipe.fit(X_train, y_train)
    svm_preds = svm_pipe.predict(X_test)
    print("SVM Pipeline Macro F1:", f1_score(y_test, svm_preds, average='macro'))
    # set zero_division=0 to avoid UndefinedMetricWarning
    print("SVM Classification Report:\n", classification_report(y_test, svm_preds, zero_division=0))

# passing split the raw text and labels for pipeline usage par D and E 
X = df['speech']
y = df['party']
X_train_text, X_test_text, y_train, y_test = splitdf(X, y)




# ========== PART D: N-gram Features (Unigrams, Bigrams, Trigrams) ==========
'''D) Use ngram_range=(1,3), max_features=3000, print classification report.'''

print("\n--- Part D: ngram_range (1,3) (Using unigrams, bigrams, and trigrams as features) ---\n")  #added ngram_range(1,3) parameter to the vectorization step in the pipe_train_model func.
pipe_train_model(X_train_text, X_test_text, y_train, y_test, ngram_range=(1, 3))

# ========== PART E: Custom Tokenizer (Fast Regex+Stopwords) ==========
'''E) Use custom tokenizer, max_features=3000, print best classifier report.'''

print("\n--- Part E: Custom Tokenizer (Using Spacy) ---\n")
pipe_train_model(X_train_text, X_test_text, y_train, y_test, ngram_range=(1, 2), tokenizer=custom_tokenizer)

