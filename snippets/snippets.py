# import pickle

# words = {'word':[], 'translation':[], 'english_pos':[], 'pos':[]}
# word_list = nw_ww.Lemma.values
# pos_list = nw_ww['POS'].values
# for n, (word, pos) in enumerate(zip(word_list, pos_list)):
#     words['word'].append(word)
#     translation = translate_txt(word, '')
#     words['translation'].append(translation)
#     words['pos'].append(pos)
#     words['english_pos'].append([x.pos() for x in wn.synsets(translation)])
#     if n % 100 == 0:
#         print((n/(len(word_list))))

# with open('data/nw_ww.pkl', 'wb') as file:
#     pickle.dump(pd.DataFrame(words), file)

# IMPORTANT SNIPPET
# verbs = {'verb':[], 'translation':[], 'english_pos':[]}
# for word in nw_ww_VC[nw_ww_VC > 1].index:
#     if word[-2:] == 'en':
#         verbs['verb'].append(word)
#         translation = translate_txt(word, '')
#         verbs['translation'].append(translation)
#         verbs['english_pos'].append([x.pos() for x in wn.synsets(translation)])
