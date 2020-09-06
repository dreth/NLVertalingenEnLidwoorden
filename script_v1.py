# %% imports
import pandas as pd
import numpy as np
from google.cloud import translate
from sklearn.preprocessing import MinMaxScaler
from json import load
from os import environ
from nltk.corpus import wordnet as wn
import spacy
from PyDictionary import PyDictionary
from collections import Counter

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


def translate_txt(text, project_id, batch=False, source="nl", target="en-US"):
    """
    Translating Text using google API
    if batch=True, text must be a list with max 1024 elements
    if batch=False, text must be a string
    """

    client = translate.TranslationServiceClient()
    params = {
        'parent': f'projects/{project_id}',
        'mime_type': "text/plain",
        'source_language_code': source,
        'target_language_code': target
    }

    if batch == False:
        params['contents'] = [text]
        response = client.translate_text(**params)
        return response.translations[0].translated_text

    else:
        if len(text) <= 1024:
            params['contents'] = text
            response = client.translate_text(**params)
            return [response.translations[x].translated_text for x in range(len(response.translations))]
        else:
            result = []
            word_lists = np.array_split(np.array(text), int(len(text)/1024)+1)
            for word_list in word_lists:
                params['contents'] = word_list
                response = client.translate_text(**params)
                response = [response.translations[x].translated_text for x in range(
                    len(response.translations))]
                for word in response:
                    result.append(word)
            return result


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
subtlexnl = pd.read_csv(
    'data/subtlex-nl-v1.3/SUBTLEX-NL.cd-above2.txt', sep='\t')
wordlexnl1 = pd.read_csv('data/wordlex-nl/Nl.Freq.2.txt', sep='\t')
wordlexnl2 = pd.read_csv('data/wordlex-nl/Nl.Freq.3.Hun.txt', sep='\t')

# %% Manipulating data
# joining wordlexnl1 and wordlexnl2
wordlexnl = pd.merge(wordlexnl1, wordlexnl2, how='inner', on=[
                     'Word', 'BlogCDPc', 'TwitterCDPc', 'NewsCDPc'])

# relative frequency of subtlexnl
total_occ = subtlexnl['FREQcount'].sum()
subtlexnl['rel_freq_sl'] = subtlexnl['FREQcount'].apply(
    lambda x: (x/total_occ)*100)

# keeping only useful columns and renaming
cols = {
    'Word': 'word',
}
subtlexnl = subtlexnl[['Word', 'rel_freq_sl']].rename(columns=cols)
wordlexnl = wordlexnl[['Word', 'BlogCDPc',
                       'TwitterCDPc', 'NewsCDPc']].rename(columns=cols)

# assigning weights to the types of frequencies
weights = {
    'BlogCDPc': 0.8,
    'TwitterCDPc': 1,
    'NewsCDPc': 0.6,
}

# normalizing columns
for n, (col, mult) in enumerate(weights.items()):
    scaler = MinMaxScaler()
    wordlexnl[col] = scaler.fit_transform(wordlexnl[[col]])*mult
    if n == len(weights.items()):
        scaler = MinMaxScaler()
        subtlexnl['rel_freq_sl'] = scaler.fit_transform(
            subtlexnl[['rel_freq_sl']])

# dropping additional cols and creating the rel freq col
wordlexnl['rel_freq_wl'] = wordlexnl[list(weights.keys())].sum(axis=1)
wordlexnl = wordlexnl[['word', 'rel_freq_wl']]

# merging only common word entries between subtlex and wordlex
data = pd.merge(subtlexnl, wordlexnl, how='inner', on='word')
data['rel_freq'] = data['rel_freq_sl'] + data['rel_freq_wl']
data = data[['word', 'rel_freq']].sort_values('rel_freq', ascending=False)
data = data.reset_index()[['word']]

# %% Tagging part of speech
# adding tag to determine type of word in order to add articles to nouns
classif1 = pd.read_csv('data/subtlex-nl-v1.3/SUBTLEX-NL.master.txt', sep='\t')
classif2 = pd.read_csv(
    'data/subtlex-nl-v1.3/SUBTLEX-NL.master.cd-above2.txt', sep='\t')
classif = pd.concat([classif1, classif2])[['Lemma', 'POS']]
classif = classif[(classif['Lemma'] != '@') & (
    classif['Lemma'].isin(data['word']))].drop_duplicates()
classif['meaning_en'] = np.nan

# %% Part of speech classsification for reference purposes
Part_of_speech = {'N': 'Noun',
                  'SPEC': 'Special',
                  'VG': 'Conjunction',
                  'ADJ': 'Adjective',
                  'WW': 'Verb',
                  'VZ': 'Preposition',
                  'BW': 'Adverb',
                  'VNW': 'Pronoun',
                  'TW': 'Numeral',
                  'TSW': 'Interjection',
                  'LID': 'Determiner'}

# %% Filtering Articles (lidwoord)
# listing all possible articles
lidwoorden = ['de', 'het', 'een']
lidwoorden_df = classif[classif['POS'] == 'LID']
classif.loc[classif['POS'] == 'LID'] = classif[(
    classif['POS'] == 'LID') & (classif['Lemma'].isin(lidwoorden))]

# de
classif.loc[(classif['Lemma'] == 'de') & (classif['POS'] == 'LID'),
            'meaning_en'] = 'the (definite article, for de-woorden)'

# een
classif.loc[(classif['Lemma'] == 'een') & (classif['POS'] == 'LID'),
            'meaning_en'] = 'a/an (indefinite article)'

# het (also as pronoun)
classif.loc[(classif['Lemma'] == 'het') & (classif['POS'] == 'LID'),
            'meaning_en'] = 'the (definite article, for het-woorden)'
classif.loc[(classif['Lemma'] == 'het') & (
    classif['POS'] == 'VNW'), 'meaning_en'] = 'it'

# finally, replacing entries labeled as lidwoord
classif.loc[classif['Lemma'].isin(
    lidwoorden)] = classif[~classif['meaning_en'].isna()]

# cleaning up removed things
classif = classif[~(classif['POS'].isna())]
lidwoorden = classif[~classif['meaning_en'].isna()]

# %% Filtering 1 letter words
classif = classif[classif['Lemma'].isin(
    [x for x in classif['Lemma'] if len(x) > 1])]

# %% Filtering 2 and 3 letter words
# and keeping them if they have a meaning in english or a translation
# such translation must also not be identical, otherwise the word will be removed from the main list
two_letter_words = list(
    set([word for word in classif['Lemma'] if len(word) < 4]))
# batch translating to find the meanings in the translation key of the dictionary
low_length_eval = {'word': two_letter_words,
                   'translation': translate_txt(two_letter_words, project_number, batch=True)}
# finding english meanings
low_length_eval['meaning_en'] = [dictionary.meaning(
    word) for word in low_length_eval['translation']]
# converting to df and filtering words that do NOT have a meaning from pydictionary and those that are
# equal to the translation in order to make a list to remove those elements
low_length_eval = pd.DataFrame(low_length_eval)
low_length_eval_remove = low_length_eval[(low_length_eval['meaning_en'].isna())
                                         & (low_length_eval['word'] == low_length_eval['translation'])]
# removing elements by filtering these out
classif = classif[~classif['Lemma'].isin(low_length_eval_remove['word'])]

# %% Filtering words where the accent emphasizes the word and doesn't change its meaning
# checking if words include accents, if so only keep those accented words
# that are french loanwords or include the words 'privé', 'één', 'vóór'
accented_words = classif[0:0]
accents = ['á', 'é', 'í', 'ó', 'ú']
no_accents = ['a', 'e', 'i', 'o', 'u']
for letter in accents:
    accented_words = pd.concat(
        [accented_words, classif[classif['Lemma'].str.contains(letter)]])

# filtering most common accented words
accented_words = accented_words[~((accented_words['Lemma'].str.contains('privé'))
                                  | (accented_words['Lemma'].str.contains('één'))
                                  | (accented_words['Lemma'].str.contains('vóór')))]
kept_accented_words = []
dataset_without_accented_words = classif[~classif['Lemma'].isin(
    accented_words)]['Lemma'].unique()

for word in accented_words['Lemma'].unique():
    # detecting if the word is originally french and adding to list to keep
    if detect_lang(word, project_number) == 'fr':
        kept_accented_words.append(word)
    else:
        for x, y in zip(accents, no_accents):
            if x in word:
                # replacing accent in the word with unaccented character
                new_word = word.replace(x, y)
                # if the unaccented word isnt in the list of main words without accents
                # keep it, otherwise, just ignore it
                if new_word not in dataset_without_accented_words:
                    kept_accented_words.append(word)
# listing all excluded words
excluded_accented_words = accented_words[~accented_words['Lemma'].isin(
    kept_accented_words)]['Lemma'].unique()
# filtering dataset by excluded words
classif = classif[~classif['Lemma'].isin(excluded_accented_words)]

# %% Importing  spacy's nl_core_news_lg model
sp = spacy.load('nl_core_news_lg')

# %% Filtering verbs, nouns and adjectives
# nouns, verbs and adjectives filter
nw_ww_adj = classif[classif['POS'].isin(['N', 'WW', 'ADJ'])]
nw_ww_adj_VC = nw_ww_adj['Lemma'].value_counts()

# extracting verbs only from previous filter
onr_verben = ['gaan', 'slaan', 'staan', 'zijn']
verben, bijvoeglijk_nmw = [], []
for word in nw_ww_adj_VC[nw_ww_adj_VC > 1].index:
    # if the word includes an irregular verb from the irr verb list in their last 5 chars, add to verb list
    cond = tuple((stem in word[-5:]) for stem in onr_verben)
    if any(cond):
        verben.append(word)
    # if the word ends in 'en' like most dutch verbs
    # but the word, when translated is not designated as a noun in english
    # then add to verb list
    elif word[-2:] == 'en':
        if sp(f'ik ga {word}')[2].pos_ in ['VERB', 'AUX']:
            verben.append(word)
        elif sp(f'mijn huis is {word}')[3].pos_ == 'ADJ':
            bijvoeglijk_nmw.append(word)
    # if the word is an adjective after labeling with spacy
    # then the word is added to the adjective list
    else:
        if sp(f'mijn huis is {word}')[3].pos_ == 'ADJ':
            bijvoeglijk_nmw.append(word)

# nouns, adjectives and verbs
# Nouns
zelfstandige_nmw = nw_ww_adj[(~(nw_ww_adj['Lemma'].isin(set(verben) | set(bijvoeglijk_nmw))) |
                              (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                             & (nw_ww_adj['POS'] == 'N')]

# Verbs
verben = nw_ww_adj[((nw_ww_adj['Lemma'].isin(verben)) |
                    (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                   & (nw_ww_adj['POS'] == 'WW')]

# Adjectives
bijvoeglijk_nmw = nw_ww_adj[((nw_ww_adj['Lemma'].isin(bijvoeglijk_nmw)) |
                             (nw_ww_adj['Lemma'].isin(nw_ww_adj_VC[nw_ww_adj_VC == 1].index)))
                            & (nw_ww_adj['POS'] == 'ADJ')]

# joining the 3 main groups
woorden = pd.concat([verben, zelfstandige_nmw, bijvoeglijk_nmw])

# %% Classifying other word taggings
# selecting subset of words per remaining PoS
remaining_pos_taggings = ['TW', 'VZ', 'VG', 'TSW', 'VNW', 'BW']
other_taggings = {pos: classif[classif['POS'] == pos]
                  for pos in remaining_pos_taggings}
kept_words_per_pos = {pos:[] for pos in Part_of_speech.keys()}
etc_words = []

# iterating over different remaining PoS selected above
# then iterating over the words and checking a sentence which would
# reasonably correspond with its appropriate tagging
for pos, word_list in other_taggings.items():
    for word in word_list['Lemma']:
        if pos == 'TW':
            if (sp(f'ik heb {word} huizen')[2].pos_ == 'NUM') or (sp(f'{word}')[0].pos_ == 'NUM'):
                kept_words_per_pos[pos].append(word)
        elif pos == 'VZ':
            if sp(f'ik ben {word} mijn huis')[2].pos_ == 'ADP':
                kept_words_per_pos[pos].append(word)
        elif pos == 'VG':
            if sp(f'ik eet {word} loop')[2].pos_ in ['CCONJ', 'SCONJ', 'CONJ']:
                kept_words_per_pos[pos].append(word)
        elif pos == 'TSM':
            if sp(f'{word}')[0].pos_ == 'INTJ':
                kept_words_per_pos[pos].append(word)
        elif pos == 'VNW':
            if (sp(f'{word} eet')[0].pos_ == 'PRON') or (sp(f'{word} eten')[0].pos_ == 'PRON'):
                kept_words_per_pos[pos].append(word)
        elif pos == 'BW':
            if sp(f'ik ga {word} eten')[2].pos_ == 'ADV':
                kept_words_per_pos[pos].append(word)
        else:
            etc_words.append(word)

# all words in teh kept words dict
all_words = []
for pos, l in kept_words_per_pos.items():
    for word in l:
        all_words.append(word)

# words that didn't pass the filters
etc_words = classif[((~classif['Lemma'].isin(all_words)) | (
    classif['Lemma'].isin(etc_words))) & (~classif['Lemma'].isin(woorden['Lemma']))]

# %% Analyzing the PoS of the rest of the words
# iterating over words and doing the tests for every possible PoS tagging
remaining_words = etc_words[['Lemma','POS']].values
conflicts = {'word':[], 'POS':[], 'amnt_cats':[], 'cats':[]}

# performing checks
for word,part_of_speech in remaining_words:
    checks = {'N': sp(f'{word} is rood')[0].pos_ in ['NOUN','PROPN'],
              'SPEC': False,
              'VG': sp(f'ik eet {word} loop')[2].pos_ in ['CCONJ', 'SCONJ', 'CONJ'],
              'ADJ': sp(f'mijn huis is {word}')[3].pos_ == 'ADJ',
              'WW': sp(f'ik ga {word}')[2].pos_ in ['VERB', 'AUX'],
              'VZ': sp(f'ik ben {word} mijn huis')[2].pos_ == 'ADP',
              'BW': sp(f'ik ga {word} eten')[2].pos_ == 'ADV',
              'VNW': (sp(f'{word} eet')[0].pos_ == 'PRON') or (sp(f'{word} eten')[0].pos_ == 'PRON'),
              'TW': (sp(f'ik heb {word} huizen')[2].pos_ == 'NUM') or (sp(f'{word}')[0].pos_ == 'NUM'),
              'TSW': sp(f'{word}')[0].pos_ == 'INTJ',
              'LID': False}
    tags = list(checks.keys())
    bools = list(checks.values())
    amnt_true = Counter([x for x in checks.values()])[True]
    if amnt_true == 1:
        kept_words_per_pos[tags[bools.index(True)]].append(word)
    else:
        all_cats = [pos for pos,b in checks.items() if b==True]
        if 'N' in all_cats:
            kept_words_per_pos['N'].append(word)
        elif 'ADJ' in all_cats:
            kept_words_per_pos['ADJ'].append(word)
        elif 'BW' in all_cats:
            kept_words_per_pos['BW'].append(word)
        else:
            conflicts['word'].append(word)
            conflicts['POS'].append(part_of_speech)
            conflicts['amnt_cats'].append(amnt_true)
            conflicts['cats'].append(tuple(all_cats))

# %% Dealing with conflicts

translated_words = dict(zip(conflicts['word'], translate_txt(conflicts['word'], project_number, batch=True)))
conflicts['en_translation'] = [translated_words[x] for x in conflicts['word']]
conflicts['pydictionary_en_pos'] = []

for word in conflicts['en_translation']:
    try:
        conflicts['pydictionary_en_pos'].append(list(dictionary.meaning(word).keys()))
    except:
        conflicts['pydictionary_en_pos'].append([])



# %% Finalizing each PoS dataframe

# zelfstandige naamwoorden, bijvoeglijk naamwoorden and verben
# defined in lines 276, 281 and 286 respectively

# Lidwoorden
# defined in line 179


# joining all word types
woorden = pd.concat([woorden, lidwoorden])
