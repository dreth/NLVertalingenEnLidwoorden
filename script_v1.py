import pandas as pd
import numpy as np
from google.cloud import translate
from sklearn.preprocessing import MinMaxScaler
from json import load
from os import environ
from nltk.corpus import wordnet as wn
import spacy
from PyDictionary import PyDictionary

# %% Basic things
# creating dictionary instance
dictionary = PyDictionary()

# google cloud authentication and client connection
environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'api-key.json'

# project number (google cloud project)
# I suggest you add this as an entry to your api key json
with open('api-key.json') as apikey:
    project_number = load(apikey)['project_number']

# %% Functions defined to translate and detect language
# using google api
def translate_txt(text, project_id, batch=False):
    """
    Translating Text using google API
    if batch=True, text must be a list
    if batch=False, text must be a string
    """

    client = translate.TranslationServiceClient()
    params = {
        'parent':f'projects/{project_id}',
        'mime_type':"text/plain",
        'source_language_code':"nl",
        'target_language_code':"en-US"
    }

    if batch == False:
        params['contents'] = [text]
        response = client.translate_text(**params)
        return response.translations[0].translated_text

    else:
        params['contents'] = text
        response = client.translate_text(**params)
        return [response.translations[x].translated_text for x in range(len(response.translations))]
    
def detect_lang(text, project_id):
    """Translating Text using google API"""

    client = translate.TranslationServiceClient()

    response = client.detect_language(
        parent=f'projects/{project_id}',
        content=text,
        mime_type="text/plain"
    )
    
    return response.languages[0].language_code


# %% Importing data
subtlexnl = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.cd-above2.txt', sep='\t')
wordlexnl1 = pd.read_csv('data/wordlex-nl/Nl.Freq.2.txt', sep='\t')
wordlexnl2 = pd.read_csv('data/wordlex-nl/Nl.Freq.3.Hun.txt', sep='\t')

# %% Manipulating data
# joining wordlexnl1 and wordlexnl2
wordlexnl = pd.merge(wordlexnl1, wordlexnl2, how='inner', on=['Word','BlogCDPc', 'TwitterCDPc', 'NewsCDPc'])

# relative frequency of subtlexnl
total_occ = subtlexnl['FREQcount'].sum()
subtlexnl['rel_freq_sl'] = subtlexnl['FREQcount'].apply(lambda x: (x/total_occ)*100)

# keeping only useful columns and renaming
cols = {
    'Word':'word',
}
subtlexnl = subtlexnl[['Word','rel_freq_sl']].rename(columns=cols)
wordlexnl = wordlexnl[['Word','BlogCDPc', 'TwitterCDPc', 'NewsCDPc']].rename(columns=cols)

# assigning weights to the types of frequencies
weights = {
    'BlogCDPc':0.8,
    'TwitterCDPc':1,
    'NewsCDPc':0.6,
}

# normalizing columns 
for n, (col, mult) in enumerate(weights.items()):
    scaler = MinMaxScaler()
    wordlexnl[col] = scaler.fit_transform(wordlexnl[[col]])*mult
    if n == len(weights.items()):
        scaler = MinMaxScaler()
        subtlexnl['rel_freq_sl'] = scaler.fit_transform(subtlexnl[['rel_freq_sl']])

# dropping additional cols and creating the rel freq col
wordlexnl['rel_freq_wl'] = wordlexnl[list(weights.keys())].sum(axis=1)
wordlexnl = wordlexnl[['word', 'rel_freq_wl']]

# merging only common word entries between subtlex and wordlex
data = pd.merge(subtlexnl, wordlexnl, how='inner', on='word')
data['rel_freq'] = data['rel_freq_sl'] + data['rel_freq_wl']
data = data[['word','rel_freq']].sort_values('rel_freq', ascending=False)
data = data.reset_index()[['word']]

# %% Tagging part of speech
# adding tag to determine type of word in order to add articles to nouns
classif1 = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.master.txt', sep='\t')
classif2 = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.master.cd-above2.txt', sep='\t')
classif = pd.concat([classif1,classif2])[['Lemma','POS']]
classif = classif[(classif['Lemma'] != '@') & (classif['Lemma'].isin(data['word']))].drop_duplicates()
classif['meaning_en'] = np.nan

# %% Filtering Articles (lidwoord)
# listing all possible articles
lidwoorden = ['de', 'het', 'een']
lidwoorden_df = classif[classif['POS'] == 'LID']
classif.loc[classif['POS'] == 'LID'] = classif[(classif['POS'] == 'LID') & (classif['Lemma'].isin(lidwoorden))]

# de
classif.loc[(classif['Lemma'] == 'de') & (classif['POS'] == 'LID'), 'meaning_en'] = 'the (definite article, for de-woorden)'

# een
classif.loc[(classif['Lemma'] == 'een') & (classif['POS'] == 'LID'), 'meaning_en'] = 'a/an (indefinite article)'

# het (also as pronoun)
classif.loc[(classif['Lemma'] == 'het') & (classif['POS'] == 'LID'), 'meaning_en'] = 'the (definite article, for het-woorden)'
classif.loc[(classif['Lemma'] == 'het') & (classif['POS'] == 'VNW'), 'meaning_en'] = 'it'

# finally, replacing entries labeled as lidwoord 
classif.loc[classif['Lemma'].isin(lidwoorden)] = classif[~classif['meaning_en'].isna()]

# %% Filtering 1 letter words
classif = classif[classif['Lemma'].isin([x for x in classif['Lemma'] if len(x) > 1])]

# %% Filtering 2 and 3 letter words
# and keeping them if they have a meaning in english or a translation
# such translation must also not be identical, otherwise the word will be removed from the main list
two_letter_words = list(set([word for word in classif['Lemma'] if len(word) < 4]))
# batch translating to find the meanings in the translation key of the dictionary
low_length_eval = {'word':two_letter_words, 
                    'translation':translate_txt(two_letter_words, project_number, batch=True)} 
# finding english meanings
low_length_eval['meaning_en'] = [dictionary.meaning(word) for word in low_length_eval['translation']]
# converting to df and filtering words that do NOT have a meaning from pydictionary and those that are
# equal to the translation in order to make a list to remove those elements
low_length_eval = pd.DataFrame(low_length_eval)
low_length_eval_remove = low_length_eval[(low_length_eval['meaning_en'].isna())
                                            & (low_length_eval['word'] == low_length_eval['translation'])]
# removing elements by filtering these out
classif = classif[~classif['Lemma'].isin(low_length_eval_remove['word'])]

# %% Filtering accented words that just emphasize and don't have a different meaning
# checking if words include accents, if so only keep those accented words
# that are french loanwords or include the words 'privé', 'één', 'vóór'
accented_words = classif[0:0]
accents = ['á', 'é', 'í', 'ó', 'ú']
no_accents = ['a', 'e', 'i', 'o', 'u']
for letter in accents:
    accented_words = pd.concat([accented_words, classif[classif['Lemma'].str.contains(letter)]])

# filtering most common accented words
accented_words = accented_words[~((accented_words['Lemma'].str.contains('privé'))
                                    | (accented_words['Lemma'].str.contains('één'))
                                    | (accented_words['Lemma'].str.contains('vóór')))]
kept_accented_words = []
dataset_without_accented_words = classif[~classif['Lemma'].isin(accented_words)]['Lemma'].unique()

for word in accented_words['Lemma'].unique():
    # detecting if the word is originally french and adding to list to keep
    if detect_lang(word, project_number) == 'fr':
        kept_accented_words.append(word)
    else:
        for x,y in zip(accents, no_accents):
            if x in word:
                # replacing accent in the word with unaccented character
                new_word = word.replace(x, y)
                # if the unaccented word isnt in the list of main words without accents
                # keep it, otherwise, just ignore it
                if new_word not in dataset_without_accented_words:
                    kept_accented_words.append(word)
# listing all excluded words
excluded_accented_words = accented_words[~accented_words['Lemma'].isin(kept_accented_words)]['Lemma'].unique()
# filtering dataset by excluded words
classif = classif[~classif['Lemma'].isin(excluded_accented_words)]

# %% Importing  spacy's nl_core_news_lg model
sp = spacy.load('nl_core_news_lg')

# %% Filtering verbs, nouns and adjectives
# nouns, verbs and adjectives filter
nw_ww_adj = classif[classif['POS'].isin(['N','WW','ADJ'])]
nw_ww_adj_VC = nw_ww_adj['Lemma'].value_counts()

# extracting verbs only from previous filter
onr_verben = ['gaan', 'slaan', 'staan', 'zijn']
verben, adjectieven = [], []
for word in nw_ww_adj_VC[nw_ww_adj_VC > 1].index:
    # if the word includes an irregular verb from the irr verb list in their last 5 chars, add to verb list
    cond = tuple((stem in word[-5:]) for stem in onr_verben)
    if any(cond):
        verben.append(word)
    # if the word ends in 'en' like most dutch verben
    # but the word, when translated is not designated as a noun in english
    # then add to verb list
    elif word[-2:] == 'en':
        zin = sp(f'ik ga {word}')
        zin2 = sp(f'mijn huis is {word}')
        if zin[2].pos_ == 'VERB':
            verben.append(word)
        elif zin2[3].pos_ == 'ADJ':
            adjectieven.append(word)
    # if the word is an adjective after labeling with spacy
    # then the word is added to the adjective list
    else:
        zin2 = sp(f'mijn huis is {word}')
        if zin2[3].pos_ == 'ADJ':
            adjectieven.append(word)

# Nouns
naamwoorden = nw_ww_adj[(~(nw_ww_adj['Lemma'].isin(verben)) |
                (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                & (nw_ww_adj['POS'] == 'N')]

# Verbs
verben = nw_ww_adj[((nw_ww_adj['Lemma'].isin(verben)) | 
                (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                & (nw_ww_adj['POS'] == 'WW')]

# Adjectives
adjectieven = nw_ww_adj[((nw_ww_adj['Lemma'].isin(adjectieven)) | 
                (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                & (nw_ww_adj['POS'] == 'ADJ')]

# joining all word types
woorden = pd.concat([verben, naamwoorden, adjectieven])



# other = classif[~classif['Lemma'].isin(nouns['Lemma'])]
# pos_count = other['Lemma'].value_counts()

# one_count = other[other['Lemma'].isin(pos_count[pos_count == 1].index)]
# main = nouns.append(one_count)

# 


