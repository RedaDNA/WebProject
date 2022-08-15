import pandas as pd
text = ["Pilots follow monarchs on the move from Canada to Mexico. Each fall, the skies between eastern Canada and the mountains of central Mexico fill with orange-and-black monarch butterflies. Along their journey, millions of butterflies fight fierce winds and flee predators. Once they reach their destination, monarchs cover trees with their bright, shimmering wings. This year, the monarchs will not be flying alone. Butterfly lovers are accompanying them on their 3,415-mile journey. They are riding in a light aircraft called a. Their aircraft is painted to resemble the orange, black, and white wings of the monarch. or move from one place to another. Beginning in early August, the butterflies head south for the winter months. Most monarchs travel from Canada to Mexico. Others travel to Florida, California, and Texas. In the spring, the butterflies migrate north. They are the only insects to make such a long round-trip migration. Mexican pilot Francisco Gutierrez and a crew of other pilots from Canada and the United States departed from Quebec, Canada, on August 15. The group plans to arrive in Mexico on November 2. The crew hopes to raise awareness about the monarchs' fragile habitats. Illegal logging in Mexico is thinning the forests that protect the butterflies from rain and cold. No one has ever followed the butterflies in the air for their entire journey. Doing so can teach scientists how the winged insects deal with wind patterns and difficult weather. Gutierrez says, It's going to be an adventure.."]
df = pd.DataFrame(text)
print(df)
import pandas as pd
import numpy as np
# for tokenization
import spacy



import pyphen

SPACY_MODEL = "en_core_web_sm"


# WORDS AND SENTENCES


def _get_words(x):
    words = [token.text for token in x if token.is_punct != True]
    return words


def words_and_sentences(df):
    """
    Uses spacy to find number of words and sentences for each text.

    Adds features:
    Avg_words_per_sentence: average number of words per sentence
    """

    # load spacy model
    nlp = spacy.load(SPACY_MODEL)

    # get tokens
    df['Tokens'] = df['Text'].apply(lambda x: nlp(x))

    # get words
    df['Words'] = df['Tokens'].apply(_get_words)

    # get sentences
    df['Sentences'] = df['Tokens'].apply(lambda x: list(x.sents))

    # get number of words
    df['N_words'] = df['Words'].apply(lambda x: len(x))

    # get number of sentences
    df['N_sentences'] = df['Sentences'].apply(lambda x: len(x))

    # also get average word number per sentence
    df["Avg_words_per_sentence"] = df["N_words"] / df["N_sentences"]

    return df


# SYLLABLES


def _count_hyphens(text, dic):
    return dic.inserted(text).count("-")


def syllables(df):
    """
    Get total number of syllables in text for each text.

    Adds features:
          Avg_syllables_per_word: average number of syllables per word

    """

    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')

    # use pyphen to find the number of hyphens (example: sentence -> sent-ence, 1 hyphen)
    df["N_hyphens"] = df["Text"].apply(lambda x: _count_hyphens(x, dic))

    # number of syllables is number of hyphens + number of words
    # (example: sentence -> sent-ence = 1 hyphen + 1 word = 2 syllables)
    df["N_syllables"] = df["N_words"] + df["N_hyphens"]

    # also write average syllable number per word
    df["Avg_syllables_per_word"] = df["N_syllables"] / df["N_words"]

    # we don't need the number of hyphens anymore
    df.drop(columns=["N_hyphens"], inplace=True)

    return df



def _get_dale_chall_easy_words():
    easy_words = set()

    with open("DEMOPROJECT/dale_chall_easy_word_list.txt") as file:
        lines = [line.rstrip('\n') for line in file]

        for line in lines:
            easy_words.add(line.lower())

    return easy_words


def _get_num_difficult_words(text, easy_words):
    n = 0
    for word in text:
        if word.lower() not in easy_words:
            n += 1
    return n


def difficult_words_pct(df):
    """
    Get percentage of difficult words as required for Dale-Chall formula.

    Adds features:
    Difficult_word_percent - percentage of difficult words (Dale-Chall)
    """

    easy_words = _get_dale_chall_easy_words()

    df["Difficult_word_percent"] = df["Words"].apply(lambda x: _get_num_difficult_words(x, easy_words)) / df["N_words"]

    return df


# POLYSYLLABLES (WORDS WITH 3 OR MORE SYLLABLES)


def _count_polysyllables(words, dic):
    n_complex = 0

    for word in words:
        # if the word has more than 3 or more syllables it will have 2 or more hyphens
        if dic.inserted(word).count("-") >= 2:
            n_complex += 1

    return n_complex


def polysyllables(df):
    """
    Get total number of polysyllables in text for each text.
    A polysyllable is a word with 3 or more syllables.

    """

    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')

    # use pyphen to find the number of polysyllables
    df["N_polysyllables"] = df["Words"].apply(lambda x: _count_polysyllables(x, dic))

    return df



def complex_words_pct(df):
    """
    Get percentage of complex words as defined by Gunning.
    Complex words (or polysyllables) are those with three or more syllables.
    Adds features:
    Complex_word_percent: percentage of complex words (Gunning)
    """

    # get percentage
    df["Complex_word_percent"] = df["N_polysyllables"] / df["N_words"]

    return df


# PERCENTAGE OF LONG SENTENCES (LONGER THAN 25 WORDS)


def _get_n_long_sent(sentences):
    n = 0
    for sentence in sentences:
        if len(sentence) > 25:
            n += 1
    return n


def long_sent_pct(df):
    """
    Get percentage of long sentences.
    Long sentences are defined as having more than 25 words.

    Adds features: Long_sent_percent: percentage of long sentences

    """

    # get percentage
    df["Long_sent_percent"] = df["Sentences"].apply(_get_n_long_sent) / df["N_sentences"]

    return df




def _get_n_long_word(words):
    n = 0
    for word in words:
        if len(word) > 8:
            n += 1
    return n


def long_word_pct(df):
    """
    Get percentage of long words.
       Long words are defined as having more than 8 chars.
    Adds features:
    Long_word_percent: percentage of long words
    """

    # get percentage
    df["Long_word_percent"] = df["Words"].apply(_get_n_long_word) / df["N_words"]

    return df



def _get_n_letters(words):
    n = 0
    for word in words:
        n += len(word)
    return n


def avg_letters_per_word(df):
    """
    Get average number of letters per word.


    Adds features:Avg_letters_per_word
    """

    # get percentage
    df["Avg_letters_per_word"] = df["Words"].apply(_get_n_letters) / df["N_words"]

    return df


def _get_n_comma_sent(sentences):
    n = 0
    for sentence in sentences:
        if str(sentence).find(",") != -1:
            n += 1
    return n


def comma_pct(df):
    """
    Get percentage of sentences with a comma.

    Adds features:
    Comma_percent: percentage of sentences with a comma
    """

    # get percentage
    df["Comma_percent"] = df["Sentences"].apply(_get_n_comma_sent) / df["N_sentences"]

    return df


def _get_n_pos(tokens, pos_list):
    n = 0
    for token in tokens:
        for pos in pos_list:
            if token.pos_ == pos:
                n += 1
    return n


def pos_features(df):
    """
    Gets several part-of-speech features:
    1) Percentage of nouns and proper nouns.
    2) Percentage of proper nouns
    3) Percentage of pronouns
    4) Percentage of conjunctions



    Adds features  Noun_percent: percentage of nouns and proper nouns
    Proper_noun_percent: percentage of proper nouns
    Pronoun_percent: percentage of pronouns
    Conj_percent: percentage of conjunctions

  -
    """

    # nouns + proper nouns percentage
    pos_list = ["NOUN", "PROPN"]
    df["Noun_percent"] = df["Tokens"].apply(lambda x: _get_n_pos(x, pos_list)) / df["N_words"]

    # proper nouns percentage
    pos_list = ["PROPN"]
    df["Proper_noun_percent"] = df["Tokens"].apply(lambda x: _get_n_pos(x, pos_list)) / df["N_words"]

    # pronouns percentage
    pos_list = ["PRON"]
    df["Pronoun_percent"] = df["Tokens"].apply(lambda x: _get_n_pos(x, pos_list)) / df["N_words"]

    # conjunctions percentage
    pos_list = ["CONJ", "CCONJ"]
    df["Conj_percent"] = df["Tokens"].apply(lambda x: _get_n_pos(x, pos_list)) / df["N_words"]

    return df

def remove_aux_features(df):

    df.drop(columns=["Tokens", "Words", "Sentences", "N_words", "N_sentences", "N_syllables", "N_polysyllables"],
            inplace=True)

    return df


from collections import Counter, defaultdict
import pandas as pd
import spacy
# benepar dependency
import benepar
from benepar.spacy_plugin import BeneparComponent


SPACY_MODEL = "en_core_web_sm"


BENEPAR_MODEL = "benepar_en3"

def _parse_tree_height(sent):
    """
    Gets the height of the parse tree for a sentence.
    """
    children = list(sent._.children)
    if not children:
        return 0
    else:
        return max(_parse_tree_height(child) for child in children) + 1


def _get_constituents(tokens):
    """
    Gets the number and average length of each constituent
    """

    const_counter = Counter()
    const_lengths = defaultdict(list)

    for sentence in tokens.sents:
        for const in sentence._.constituents:
            # add constituent to constituent counter
            const_counter.update(Counter(const._.labels))

            # append the length of the constituent
            for label in const._.labels:
                const_lengths[label].append(len(const))

    # for each constituent, get average of constituent's lengths
    const_avgs = defaultdict(int)
    for key in const_lengths.keys():
        avg = 0.0
        for length in const_lengths[key]:
            avg += length
        avg /= len(const_lengths[key])

        const_avgs[key] = avg

    return const_counter, const_avgs


def _get_parse_tree_height(tokens):
    """
    Get averagee  parse tree height of each  sentence
    """
    avg_parse_tree_height = 0.0

    for sentence in tokens.sents:
        avg_parse_tree_height += _parse_tree_height(sentence)

    n_sentences = len(list(tokens.sents))
    avg_parse_tree_height /= n_sentences

    return avg_parse_tree_height, n_sentences


def _get_parse_tree_features(tokens):
    const_counter, const_avgs = _get_constituents(tokens)
    avg_parse_tree_height, n_sentences = _get_parse_tree_height(tokens)

    NP_per_sent = const_counter['NP'] / n_sentences
    VP_per_sent = const_counter['VP'] / n_sentences
    PP_per_sent = const_counter['PP'] / n_sentences
    SBAR_per_sent = const_counter['SBAR'] / n_sentences
    SBARQ_per_sent = const_counter['SBARQ'] / n_sentences
    avg_NP_size = const_avgs['NP']
    avg_VP_size = const_avgs['VP']
    avg_PP_size = const_avgs['PP']
    avg_parse_tree = avg_parse_tree_height

    return NP_per_sent, VP_per_sent, PP_per_sent, \
           SBAR_per_sent, SBARQ_per_sent, avg_NP_size, \
           avg_VP_size, avg_PP_size, avg_parse_tree


def parse_tree_features(df):
    """
    Adds features:     NP_per_sent: NPs (noun phrase) / num of sentences
    VP_per_sent: VPs (verb phrase) / num of sentences
    PP_per_sent: PPs (prepositional phrase) / num of sentences
    SBAR_per_sent: SBARs (subordinate clause) / num of sentences
    SBARQ_per_sent: SBARQs (direct question introduced by wh-element) / num of sentences
    avg_NP_size: Average lenght of an NP
    avg_VP_size: Average lenght of an VP
    avg_PP_size: Average lenght of an PP
    avg_parse_tree: Average height of a parse Tree
    """

    nlp = spacy.load(SPACY_MODEL, disable=['ner'])
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    df['B_Tokens'] = df['Text'].apply(lambda x: nlp(x))

    # get features
    df['NP_per_sent'], df['VP_per_sent'], df['PP_per_sent'], \
    df['SBAR_per_sent'], df['SBARQ_per_sent'], df['avg_NP_size'], \
    df['avg_VP_size'], df['avg_PP_size'], df['avg_parse_tree'] = zip(*df['B_Tokens'].map(_get_parse_tree_features))

    df.drop(columns=["B_Tokens"], inplace=True)

    return df

