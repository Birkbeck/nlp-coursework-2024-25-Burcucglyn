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