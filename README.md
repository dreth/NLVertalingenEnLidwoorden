# NLVertalingenEnLidwoorden
Taking subtlex-nl and wordlex word frequency data, translating using a translator api and adding articles for nouns with the purpose of making an anki deck and a shareable dataset

## Citations
I do not own or authored subtlex-nl or wordlex!

A big thank you to the authors of these excellent papers.

* Keuleers, E., Brysbaert, M. & New, B. (2010). SUBTLEX-NL: A new frequency measure for Dutch words based on film subtitles. Behavior Research Methods, 42(3), 643-650.
  * **subtlex-nl website**: http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-nl
  * **subtlex-nl paper**: http://crr.ugent.be/papers/SUBTLEX-NL_BRM.pdf

* Gimenes, M., New, B. Worldlex: Twitter and blog word frequencies for 66 languages. Behav Res 48, 963–972 (2016). https://doi.org/10.3758/s13428-015-0621-0
  * **wordlex paper**: https://link.springer.com/article/10.3758/s13428-015-0621-0

## Methodology
1 - import libraries and define function to translate words using google cloud translate API

2 - read subtlex-nl and wordlex-nl datasets into a pandas dataframe

3 - merging the 2 wordlex nl datasets (2 are provided on the original site)

4 - rename word column to be lowercase and select df columns that are useful for our work

5 - assign weights to the different wordlex nl word sources, we assign a higher weight to twitter as it would highlight more colloquial language use (which for learning purposes is more useful for our end result)

6 - normalize columns and apply weight multiplier using MinMaxScaler from sklearn

7 - join both wordlex nl and subtlex-nl data using a sum of both relative frequencies to set an order for words

8 - import the subtlex-nl data that details the functions of words the sentences as a code

  * The information about the Part of speech/lemma types explanation for subtlex-nl can be found here: http://crr.ugent.be/archives/362

9 - given that a word can have different functions in a sentence, some words in the part of speech dataset of subtlex-nl are repeated. The focus of this project is to help dutch learners memorize useful words ordered by frequency. The first filter was applied to extract *nouns and verbs* first. The process here is the following:

  * filter nouns and verbs
  * count lemmas to see in how many categories they are, if 2 comes up, theyre classified as verb and noun
  * we extract words that end in 'gaan', 'slaan', 'staan' or 'zijn', as these verbs do not end in 'en'
  * because there are so many words ending in 'en' that are *not* verbs, we need to apply some kind of filter to fairly choose them, so what we do is
    * check if it has 'en' as ending
    * translate the word to english
    * check the function of that word in english
    * if the function is noun, we take it out of the verb list
  * add this list of verbs and nouns finally separated in a general dataframe of words excluded from the general list called nw_ww








