import pandas as pd
import numpy as np
from google.cloud import translate
from sklearn.preprocessing import MinMaxScaler
from json import load
from os import environ
from nltk.corpus import wordnet as wn


# google cloud authentication and client connection
environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'api-key.json'

# using google api
def translate_txt(text, project_id):
    """Translating Text using google API"""

    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=f'projects/{project_id}',
        contents=[text],
        mime_type="text/plain",
        source_language_code="nl",
        target_language_code="en-US",
    )
    
    return response.translations[0].translated_text


# importing data
subtlexnl = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.cd-above2.txt', sep='\t')
wordlexnl1 = pd.read_csv('data/wordlex-nl/Nl.Freq.2.txt', sep='\t')
wordlexnl2 = pd.read_csv('data/wordlex-nl/Nl.Freq.3.Hun.txt', sep='\t')

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

# adding tag to determine type of word in order to add articles to nouns
classif1 = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.master.txt', sep='\t')
classif2 = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.master.cd-above2.txt', sep='\t')
classif = pd.concat([classif1,classif2])[['Lemma','POS']]
classif = classif[(classif['Lemma'] != '@') & (classif['Lemma'].isin(data['word']))].drop_duplicates()

nw_ww = classif[classif['POS'].isin(['N','WW'])]
nw_ww_VC = nw_ww['Lemma'].value_counts()
nw_ww_main = nw_ww[nw_ww['Lemma'].isin(nw_ww_VC[nw_ww_VC == 1].index)]

# extracting verbs only from previous filter
def extract_verbs(nw_ww_value_count, project_id):
    irr_verbs = ['gaan', 'slaan', 'staan', 'zijn']
    verbs = []
    for word in nw_ww_value_count[nw_ww_value_count > 1].index:
        # if the word includes an irregular verb from the irr verb list in their last 5 chars, add to verb list
        cond = tuple((stem in word[-5:]) for stem in irr_verbs)
        if any(cond):
            verbs.append(word)
        # if the word ends in 'en' like most dutch verbs
        # but the word, when translated is not designated as a noun in english
        # then add to verb list
        elif word[-2:] == 'en':
            
            verbs.append(word)

# IMPORTANT SNIPPET
verbs = {'verb':[], 'translation':[], 'english_pos':[]}
for word in nw_ww_VC[nw_ww_VC > 1].index:
    if word[-2:] == 'en':
        verbs['verb'].append(word)
        translation = translate_txt(word, 'YOUR PROJECT ID')
        verbs['translation'].append(translation)
        verbs['english_pos'].append(wn.synsets(translation)[0].pos())

# other = classif[~classif['Lemma'].isin(nouns['Lemma'])]
# pos_count = other['Lemma'].value_counts()

# one_count = other[other['Lemma'].isin(pos_count[pos_count == 1].index)]
# main = nouns.append(one_count)

# 


# import pickle

# words = {'word':[], 'translation':[], 'english_pos':[], 'pos':[]}
# word_list = nw_ww.Lemma.values
# pos_list = nw_ww['POS'].values
# for n, (word, pos) in enumerate(zip(word_list, pos_list)):
#     words['word'].append(word)
#     translation = translate_txt(word, '1072058686454')
#     words['translation'].append(translation)
#     words['pos'].append(pos)
#     words['english_pos'].append([x.pos() for x in wn.synsets(translation)])
#     if n % 100 == 0:
#         print((n/(len(word_list))))

# with open('data/nw_ww.pkl', 'wb') as file:
#     pickle.dump(pd.DataFrame(words), file)
