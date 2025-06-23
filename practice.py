''' Part 1: 
Part One — Syntax and Style
In the first part of your coursework, your task is to explore the syntax and style of a
set of 19th Century novels using the methods and tools that you learned in class.
The texts you need for this part are in the novels subdirectory of the texts directory in
the coursework Moodle template. The texts are in plain text files, and the filenames
include the title, author, and year of publication, separated by hyphens. The template
code provided in PartOne.py includes function headers for some sub-parts of this ques-
tion. The main method of your finished script should call each of these functions in
order. To complete your coursework, complete these functions so that they perform
the tasks specified in the questions below. You may (and in some cases should) define
additional functions.
(a) 4read_novels: Each file in the novels directory contains the text of a novel, and
the name of the file is the title, author, and year of publication of the novel,
separated by hyphens. Complete the python function read_texts to do the
following:
i. create a pandas dataframe with the following columns: text, title, author,
year
ii. sort the dataframe by the year column before returning it, resetting or
ignoring the dataframe index. '''

'''Part 1. a) read_novels and file names: title, author, and year of publication of the novel,
separated by hyphens
I) import panda for dataframe following columns: text, title, author,
year '''


import pandas as pd
import os
import glob

import pandas as pd
import os
import glob

def read_novels(path):
    data = []
    files = glob.glob(os.path.join(path, '*.txt'))
    print("Found files:", files)  # Debug: show found files
    for file in files:
        filename = os.path.basename(file)
        print("Processing filename:", filename)  # Debug: show each filename
        name = filename[:-4]  # removes '.txt'
        try:
            title, author, year = name.rsplit('-', 2)
            year = int(year)
        except ValueError:
            print(f"Skipping file (bad name format): {filename}")
            continue
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        data.append({'text': content, 'title': title, 'author': author, 'year': year})
    print(f"Loaded {len(data)} files.")
    df = pd.DataFrame(data)
    print(df.columns)
    df = df.sort_values('year').reset_index(drop=True)
    return df

# Call your function with the correct path
df = read_novels("/Users/burdzhuchaglayan/Desktop/Msci_DS/summer term/natural language processing/coursework needs to be submitted at 3rd of july/p1-texts/novels")
print(df)