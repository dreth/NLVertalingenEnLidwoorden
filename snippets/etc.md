Second methodology for nouns/verbs

B - check if it has 'en' as ending
      * translate the word to english
      * look for the word in english in the NLTK WordNet
      * check all the possible PoS of this word in the sentence
      * the PoS with the most appearances will be used
      * if empty, assign as verb, if tie, assign first observed of those two most voted options