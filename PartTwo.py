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

#for custom tokenizer in E
import spacy
nlp = spacy.load("en_core_web_sm")



'''A) I. Read the hansard40000.csv dataset in the texts directory into a dataframe.
    - Change the 'Labour (Co-op)' value in the 'party' column to 'Labour'.
    - Check if the function is working by printing the first few rows.
'''

def read_hansard(path=Path.cwd() / "texts" / "hansard40000.csv"):
    """Reads the hansard40000.csv dataset"""
    df = pd.read_csv(path)
    # Change the 'Labour (Co-op)' value in 'party' column to 'Labour'
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    return df

# Check if the function is working
df = read_hansard("/Users/burdzhuchaglayan/Desktop/Msci_DS/summer term/natural language processing/coursework needs to be submitted at 3rd of july/p2-texts/hansard40000.csv")
print(df.head())

'''A) II. Remove any rows where the value of the party column is not one of the main 4 party and remove the 'Speaker' value.'''

#Print the parties in the party column to see which ones are present
#print(df['party'].unique())

''' output: ['Labour' 'Conservative' 'Liberal Democrat'
 'Speaker' 'Democratic Unionist Party' 'Independent' 
 'Social Democratic & Labour Party' 'Alliance' 'Green Party' 'Alba Party']'''

#4 main parties are (googled it) :  'Labour' ,'Conservative' , 'Liberal Democrat', 'Independent' 

def filter_parties(df):
    """Filters the dataframe to keep only the main 4 parties and removes the 'Speaker' value."""
    #to see the party column values used the print(df['party'].unique()) function
    main_parties = ['Labour', 'Conservative', 'Liberal Democrat', 'Independent']
    #remove rows where the party is not in the main 4 and remove the 'Speaker' value
    df = df[df['party'].isin(main_parties)] #isin pandas methods to filter the df 
    return df

# Check if the function is working
df = filter_parties(df)
print(df['party'].unique())  # Should only show the main 4 parties
print(df.shape)

'''A) III. Remove any rows where the value in the speech_class column is not 'Speech'.'''
def filter_speech(df):
    """Filters the dataframe to keep only rows where speech_class is 'Speech'."""
    df = df[df['speech_class'] == 'Speech']  # Keep only rows where speech_class is 'Speech'
    return df

# Check if the function is working
df = filter_speech(df)
print(df['speech_class'].unique())

'''A) IV. Remove any rows where the text in the speech column is less than 1000 characters long.'''
def filter_speech_length(df):
    """Filters the dataframe to keep only rows where the speech text is at least 1000 characters long."""
    df = df[df['speech'].str.len() >= 1000]  # Keep only rows  length of speech is at least 1000 characters
    return df

# Check if the function is working
df = filter_speech_length(df)
print(df.shape)  # Print the shape of the dataframe to see how many rows are left



''' B) Vectorise the speeches. Omitting English stopwords and setting max_features=30000. Split the data into a train and test set, using stratified sampling, with a
random_state = 26.'''

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

''''c) Train RandomForest (with n_estimators=300) and SVM with linear kernel clas-
sifiers on the training set, and print the scikit-learn macro-average f1 score and
classification report for each classifier on the test set. The label that you are
trying to predict is the ‘party’ value.'''


# def classifiers_models (X_train, X_test, y_train, y_test):
#     """
#     Trains RandomForest and SVM classifiers on the training set and prints the macro-average f1 score and classification report.
#     Already have X_train, X_test, y_train, y_test from your vectorization step, so won't split again
#     """

#     # Random Forest
#     rf_model = RandomForestClassifier(n_estimators=300, random_state=26, class_weight='balanced') #additional add for non biased model added class_weight='balanced'
#     rf_model.fit(X_train, y_train)
#     rf_predictions = rf_model.predict(X_test)
#     rf_f1_score = f1_score(y_test, rf_predictions, average='macro')
#     print("Random Forest Macro F1 Score:", rf_f1_score)
#     print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions)) 

#     # Linear SVM
#     svm_model = SVC(kernel='linear', random_state=26, class_weight='balanced') #additional add for non biased model added class_weight='balanced'
#     svm_model.fit(X_train, y_train)
#     svm_predictions = svm_model.predict(X_test)
#     svm_f1_score = f1_score(y_test, svm_predictions, average='macro')
#     print("SVM Macro F1 Score:", svm_f1_score)
#     print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# # Usage:
# classifiers_models(X_train, X_test, y_train, y_test)

# # Apply SMOTE to the training data only
# smote = SMOTE(random_state=26)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print("After SMOTE oversampling:")
# print("y_train_resampled value counts:\n", pd.Series(y_train_resampled).value_counts())

# # Now train your models on the resampled data
# classifiers_models(X_train_resampled, X_test, y_train_resampled, y_test)

''' After running the code, got warnings: UndefinedMetricWarning for some classes 
(precision and F-score set to 0.0 when no predicted samples). 
Checked and found 'Independent','Liberal Democrat' have very few samples, 
so the model ignores them and gets biased toward majority classes. 
To fix, can try class weighting in RandomForest and SVM (class_weight='balanced').
This adjusment didn't completely solve the issue, so I applied SMOTE to oversample 
the minority classes in the training set.'''

'''Smote oversampling helpted balance the classes but still some classes have low precision 
and recall. So I will try to use a different approach like using a different model. 
I will try using scikit-learn pipeline to combine the vectorization and classification steps,
and then apply SMOTE within the pipeline. This way, I can ensure that the oversampling 
is done correctly.
'''

''' Trains RandomForest and SVM classifiers using a pipeline with SMOTE for oversampling with
the training set, and prints the macro-average f1 score and classification report.'''


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

'''d) Adjust the parameters of the Tfidfvectorizer so that unigrams, bi-grams and
tri-grams will be considered as features, limiting the total number of features to
3000. Print the classification report as in 2(c) again using these parameters. '''

'''in this part I will adjust the code I created on part B and C to include unigrams, bi-grams and tri-grams as features, to order to do that i will change the ngram_range parameter of TfidfVectorizer to (1, 3) which means it will consider unigrams, bi-grams and tri-grams as features.'''

#added ngram_range(1,3) parameter to the vectorization step in the pipe_train_model function now it needs to be called again without writing a new func. 

print("\n--- Part D: ngram_range (1,3) (Using unigrams, bigrams, and trigrams as features) ---\n")
pipe_train_model(X_train, X_test, y_train, y_test, ngram_range=(1, 3))

''' E) Implement a new custom tokenizer and pass it to the tokenizer argument of
Tfidfvectorizer. You can use this function in any way you like to try to achieve
the best classification performance while keeping the number of features to no
more than 3000, and using the same three classifiers as above. Print the clas-
sification report for the best performing classifier using your tokenizer. Marks
will be awarded both for a high overall classification performance, and a good
trade-off between classification performance and efficiency (i.e., using fewer pa-
rameters).'''

def custom_tokenizer(text):
    doc = nlp(text.lower())
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

#Now I will use pipe_train_model function to train the models with the custom tokenizer
print("\n--- Part E: Custom Tokenizer (Using Spacy) ---\n")
pipe_train_model(X_train, X_test, y_train, y_test, ngram_range=(1, 2))  # Using the custom tokenizer with n-grams (1,2)

# Split the raw text and labels, not the vectorized features
X = df['speech']
y = df['party']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=26, stratify=y
)

pipe_train_model(X_train_text, X_test_text, y_train, y_test, ngram_range=(1, 3))

