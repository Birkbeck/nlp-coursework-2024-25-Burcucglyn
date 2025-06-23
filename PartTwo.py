'''Read the hansard40000.csv dataset in the texts directory into a dataframe. Sub-
set and rename the dataframe as follows:
i. rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’, and
then:
ii. remove any rows where the value of the ‘party’ column is not one of the
four most common party names, and remove the ‘Speaker’ value.
iii. remove any rows where the value in the ‘speech_class’ column is not
‘Speech’.
iv. remove any rows where the text in the ‘speech’ column is less than 1000
characters long.
Print the dimensions of the resulting dataframe using the shape method.'''

from pathlib import Path
import pandas as pd


'''A) I. Read the hansard40000.csv dataset in the texts directory into a dataframe.
    - Change the 'Labour (Co-op)' value in the 'party' column to 'Labour'.
    - Check if the function is working by printing the first few rows.
'''

def read_hansard(path=Path.cwd() / "texts" / "hansard40000.csv"):
    """Reads the hansard40000.csv dataset and processes it according to the specified criteria."""
    df = pd.read_csv(path)
    # Change the 'Labour (Co-op)' value in 'party' column to 'Labour'
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    return df

# Check if the function is working
df = read_hansard("/Users/burdzhuchaglayan/Desktop/Msci_DS/summer term/natural language processing/coursework needs to be submitted at 3rd of july/p2-texts/hansard40000.csv")
print(df.head())

