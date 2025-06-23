from pathlib import Path
import pandas as pd

#vectorization and splitting for B
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split   

#random forest and SVM for C
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC

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


def classifiers_models (X_train, X_test, y_train, y_test):
    """
    Trains RandomForest and SVM classifiers on the training set and prints the macro-average f1 score and classification report.
    Already have X_train, X_test, y_train, y_test from your vectorization step, so won't split again
    """

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=26)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_f1_score = f1_score(y_test, rf_predictions, average='macro')
    print("Random Forest Macro F1 Score:", rf_f1_score)
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions)) 

    # Linear SVM
    svm_model = SVC(kernel='linear', random_state=26)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_f1_score = f1_score(y_test, svm_predictions, average='macro')
    print("SVM Macro F1 Score:", svm_f1_score)
    print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Usage:
classifiers_models(X_train, X_test, y_train, y_test)