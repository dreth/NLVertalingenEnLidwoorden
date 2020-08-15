import pandas as pd
import numpy as np
from google.cloud import translate
from sklearn.preprocessing import MinMaxScaler
from json import load
import os
import requests
import urllib.request
from bs4 import BeautifulSoup
import googletrans

# credentials
apikey_path = 'api-key.json'
with open(apikey_path) as apikey: 
    apikey = load(apikey)

# google cloud authentication and client connection
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = apikey_path
client = translate.TranslationServiceClient()


# using google api
def translate_txt(text, project_id):
    """Translating Text using google API"""

    response = client.translate_text(
        parent=f'projects/{project_id}',
        contents=[text],
        mime_type="text/plain",
        source_language_code="nl",
        target_language_code="en-US",
    )
    
    return response


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

# normalizing cols and assigning weights to the frequencies
weights = {
    'BlogCDPc':0.8,
    'TwitterCDPc':1,
    'NewsCDPc':0.6,
}
for n, (col, mult) in enumerate(weights.items()):
    scaler = MinMaxScaler()
    wordlexnl[col] = scaler.fit_transform(wordlexnl[[col]])*mult
    if n == len(weights.items()):
        scaler = MinMaxScaler()
        subtlexnl['rel_freq_sl'] = scaler.fit_transform(subtlexnl[['rel_freq_sl']])

# dropping additional cols and creating the rel freq col
wordlexnl['rel_freq_wl'] = wordlexnl[list(weights.keys())].sum(axis=1)
wordlexnl = wordlexnl[['word', 'rel_freq_wl']]

# merging only common word entries
data = pd.merge(subtlexnl, wordlexnl, how='inner', on='word')
data['rel_freq'] = data['rel_freq_sl'] + data['rel_freq_wl']
data = data[['word','rel_freq']].sort_values('rel_freq', ascending=False)
data = data.reset_index()[['word']]

# scraping google translate site to know what type of word it is
def google_translate_scraper(word, sl, tl):
    url = f'https://translate.google.com/#view=home&op=translate&sl={sl}&tl={tl}&text={word}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup