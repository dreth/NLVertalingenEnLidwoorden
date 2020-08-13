import pandas as pd
import numpy as np
from google.cloud import translate

# using google api


# importing data
subtlexnl = pd.read_csv('subtlex-nl-v1.3/SUBTLEX-NL.cd-above2.txt', sep='\t')
wordlexnl1 = pd.read_csv('wordlex-nl/Nl.Freq.2.txt', sep='\t')
wordlexnl2 = pd.read_csv('wordlex-nl/Nl.Freq.3.Hun.txt', sep='\t')

# joining wordlexnl1 and wordlexnl2
wordlexnl = pd.merge(wordlexnl1, wordlexnl2, how='inner', on=['Word','BlogCDPc', 'TwitterCDPc', 'NewsCDPc'])

# relative frequency of subtlexnl
total_occ = subtlexnl['FREQcount'].sum()
subtlexnl['rel_freq'] = subtlexnl['FREQcount'].apply(lambda x: (x/total_occ)*100)

# keeping only useful columns and renaming
cols = {
    'Word':'word',
}
subtlexnl = subtlexnl[['Word','rel_freq']].rename(columns=cols)
wordlexnl = wordlexnl[['Word','BlogCDPc', 'TwitterCDPc', 'NewsCDPc']].rename(columns=cols)


# assigning weights to the frequencies
weights = {
    'BlogCDPc':1.5,
    'TwitterCDPc':1,
    'NewsCDPc':0.8,
}
wordlexnl['rel_freq'] = 

# merging only common word entries
data = pd.merge(subtlexnl, wordlexnl1, how='inner', on='Word')
data = data.merge(wordlexnl2, how='inner', on='Word')

