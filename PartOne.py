#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. 
#You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize
import re
# progress_apply = pd.DataFrame.progress_apply 
from tqdm import tqdm
tqdm.pandas()
# import counter for counting word frequencies
from collections import Counter



nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    
    from nltk.tokenize import sent_tokenize, word_tokenize

    #tokenize sentences and words in the text
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_alpha = [w for w in words if w.isalpha()]

    num_words = len(words_alpha)
    num_sentences = len(sentences)
    syllables = sum(count_syl(word, d) for word in words_alpha)

    if num_words == 0 or num_sentences == 0:
        return None
    # Flesch-Kincaid Grade Level formula
    fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
    return fk_grade


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word."""
    word = word.lower()
    if word in d:
        return len([ph for ph in d[word][0] if ph[-1].isdigit()])
    else:
        return len(re.findall(r'[aeiouy]+', word))

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    path = Path(path)
    files = list(path.glob('*.txt'))
    for file in files:
        filename = file.name
        name = filename[:-4]  # removes the last 4 character before '.txt'
        try:
            title, author, year = name.rsplit('-', 2)
            year = int(year)
        except ValueError:
            print(f"Skipping file (bad name format): {filename}")
            continue
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        data.append({'text': content, 'title': title, 'author': author, 'year': year})
    df = pd.DataFrame(data)
    df = df.sort_values('year').reset_index(drop=True)
    return df



def parse(df, store_path=Path.cwd() / "pickles", out_name="named.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    #used spaCy to parse the text and add a new column to the df.
    df["parsed"] = df["text"].progress_apply(nlp) #using tqdm to show progress in the notebook
    #store the DataFrame to a pickle file
    store_path.mkdir(parents=True, exist_ok=True)
    df.to_pickle(store_path / out_name)
    return df




def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    return len(set(tokens))/ len(tokens) if tokens else None
    


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        score = fk_level(row["text"], cmudict)
        results[row["title"]] = round(score, 4) if score is not None else 0
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            # Check if the token is a verb and has a subject dependency
            for child in token.children: #dependency tree token.children
                if child.dep_ in ("nsubj", "nsubjpass"): #dependency tree checks if the child is a subject or passive subject
                    # If the child is a subject, add the subject phrase to the list
                    subj_phrase = " ".join([t.text for t in child.subtree])
                    # Count the frequency of each subject phrase and return the most common ones    
                    subjects.append(subj_phrase)
    return Counter(subjects).most_common(10) # Returns the 10 most common subjects as a list of tuples (subject, count)




def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjs =[token.text for token in doc if token.pos_ == "ADJ"] #spacy tokenizationm adj checks if the token is an adjective and than gets the text of the token
    # tuples for Counting the frequency of adj.
    return Counter(adjs).most_common(10) #returns the 10 most common adjectives as a list of tuples (adj, count)




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels("/Users/burdzhuchaglayan/Desktop/Msci_DS/summer term/natural language processing/coursework needs to be submitted at 3rd of july/p1-texts/novels") # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    nltk.download("cmudict")
    # parse(df, out_name="parsed.pickle")  # this line will fail until you have completed the parse function above.
    
   # print(get_ttrs(df))
   # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    # print(df.head())
    # print(adjective_counts(df))

    #test adjective_counts
    # for i, row in df.iterrows():
    #     print(row["title"])
    #     print(adjective_counts(row["parsed"]))
    #     print("\n")
    
    #test subjects_by_verb_count
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))  # Change "hear" to any verb you want to test
        print("\n")

    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"]))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

