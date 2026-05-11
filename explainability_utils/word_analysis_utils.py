import re

import nltk
import simple_icd_10_cm as icd10_lookup_cm
#from cie.cie10 import CIECodes
from nltk.util import ngrams
from ipymarkup import show_span_box_markup


def find_all_word_occurrences(text, word):
    """
    Given a text and a word, return a list of tuples with the start and end of each occurrence of the word in the text
    :param text: text
    :param word: word
    :return: list of tuples with the start and end of each occurrence of the word in the text
    """
    occurrences = []
    if len(word)>0:
        for match in re.finditer(r'\b{}\b'.format(re.escape(word)), text, flags=re.IGNORECASE):
            occurrences.append((match.start(), match.end()))
    return occurrences


def find_all_grams_occurrences(text, sentence):
    """
    Given a text and a subtext, return a list of tuples with the start and end of each occurrence of the subsentence in the text
    :param text: text
    :param word: word
    :return: list of tuples with the start and end of each occurrence of the word in the text
    """
    occurrences = []
    if len(sentence)>0:
        pattern = re.compile(re.escape(sentence))
        for match in pattern.finditer(text):
            occurrences.append((match.start(), match.end()))
    return occurrences

def get_top_word_weights(attributions_dict, top_n=10, clean_stopwords=False, language="english"):
    """
    Given a dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label
    and the number of words to highlight, return the top n words and their attributions
    :param attributions_dict: dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label
    :param top_n: number of words to highlight
    :param clean_stopwords: if True, remove stopwords
    :param language: language of the text

    return: dictionary with the top n words and their attributions
    """
    top_word_attributions_dict={}
    for key in attributions_dict.keys():
        attributions=attributions_dict[key]
        # Get the words with the highest attribution
        highligth_words=get_top_n_words(attributions, top_n=top_n, clean_stopwords=clean_stopwords, language=language)
        filtered_attributions={word:attributions[word] for word in highligth_words}
        top_word_attributions_dict[key]=filtered_attributions
    return top_word_attributions_dict

def get_top_n_grams_weights(token_list,attributions_dict,top_word_weights, n=4):
    """
    Given a token_list,a dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label,
    the attributions of the top words and the number of words in the n-gram, return the top n n-grams and their attributions
    :param token_list: list of tokens
    :param attributions_dict: dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label
    :param top_word_weights: dictionary of list of tuples with the attributions of each top word (word, attribution) for each label
    :param n: number of words in the n-gram

    return: dictionary with the top n n-grams and their attributions
    """
    top_n_grams_dict={}
    for key in attributions_dict.keys():
        attributions = attributions_dict[key]
        # Get the words with the highest attribution
        highligth_words = list(top_word_weights[key].keys())
        # Get the n-grams with the highest attribution
        top_n_grams = get_top_n_grams(token_list, attributions, highligth_words, n=n,return_weights=True)
        top_n_grams_dict[key]=top_n_grams
    return top_n_grams_dict


def generate_spans(labels_tokens_dict):
    """
    Given a list of list with labels and tokens index, and a text, return a list of tuples with the start and end of each span
    and the label
    :param labels_tokens_dict: list of list with labels and tokens index
    :param text: text
    :return: html with colored text
    """
    spans=[]
    for label in labels_tokens_dict:
        for token in labels_tokens_dict[label]:
            if not token[0] == -1:
                spans.append((token[0],token[-1],label))
    return spans


def generate_colored_text(labels_tokens_dict,text):
    """
    Given a list of list with labels and tokens index, and a text, return a list of tuples with the start and end of each span
    and the label
    :param labels_tokens_dict: list of list with labels and tokens index
    :param text: text
    :return: html with colored text
    """
    spans=generate_spans(labels_tokens_dict)
    print(spans)
    html = show_span_box_markup(text, spans)
    print(html)
    return html

def get_top_html(token_list,top_word_weights,by_label=False,n_grams=False):
    """
    Given a token_list and the attributions of the top words, return the html with the highlighted words
    :param token_list: list of tokens
    :param top_word_weights: dictionary of list of tuples with the attributions of each top word (word, attribution) for each label
    :param by_label: if True, return the html by label
    :param n_grams: if True, return the html by n-grams

    return: html with the highlighted words
    """
    text=" ".join(token_list)
    label_words_index = {}
    html=""
    for key in top_word_weights.keys():
        attributions=top_word_weights[key]
        # Get the words with the highest attribution
        # print(attributions)
        words_index = get_words_indexes(text, list(attributions.keys()), ngrams=n_grams)
        print("WORDS INDEX")
        print(words_index)
        label_words_index[key] = words_index
        if by_label:
            html+="\n <h4>"+key.upper()+"</h4>\n"
            print({key:words_index})
            current_html= generate_colored_text({key:words_index}, text)
            if current_html is None:
                print("current_html es None aquí, revisa por qué")
            else:
                html += current_html.data
            label_words_index = {}
    if not by_label:
        html = generate_colored_text(label_words_index, text)
        html = html.data

    return html

def get_top_html_by_label_dict(token_list,top_word_weights,n_grams=False):
    # Reconstruct words from tokens by removing ## separators
    clean_token_list = []
    current_word = ""
    for token in token_list:
        if token.startswith("##"):
            # This is a continuation of the previous word (subword token)
            current_word += token.replace("##", "")
        else:
            # This is the start of a new word
            if current_word:
                clean_token_list.append(current_word)
            current_word = token
    # Add the last word
    if current_word:
        clean_token_list.append(current_word)
    
    text = " ".join(clean_token_list)
    label_words_index = {}
    html_dict = {}
    for key in top_word_weights.keys():
        attributions=top_word_weights[key]
        # Get the words with the highest attribution
        # Clean the attribution keys (remove ## if present)
        cleaned_attribution_keys = []
        for word in attributions.keys():
            # Remove ## from word if present
            cleaned_word = word.replace("##", "")
            cleaned_attribution_keys.append(cleaned_word)
        
        words_index = get_words_indexes(text, cleaned_attribution_keys, ngrams=n_grams)
        print("WORDS INDEX")
        print(words_index)
        label_words_index[key] = words_index
        html=""
        print({key:words_index})
        current_html= generate_colored_text({key:words_index}, text)
        if current_html is None:
            print("current_html es None aquí, revisa por qué")
        else:
            html += current_html.data

        html_dict[key] = html
        label_words_index = {}
    return html_dict

def get_top_n_words(attributions_dict,clean_stopwords=False,language="english",top_n=10):
    """
    Given a dictionary with the attributions of each word or n-gram {label:(word,attribution)}, return the top n words
    and their attributions
    :param attributions_dict: dictionary with the attributions of each word or n-gram {label:(word,attribution)}
    :param clean_stopwords: if True, remove stopwords
    :param language: language of the text
    :param top_n: number of words to highlight

    return: dictionary with the top n words and their attributions
    """
    # Sort the attributions dict by value

    top_n_words = sorted(attributions_dict.items(), key=lambda x: x[1], reverse=True)

    if len(attributions_dict.items()) > top_n:
        # Get the words with the highest attribution
        # Sort the attributions list by the second element of the tuple (the attribution)
        # Check whether it is a list of tuples or a dictionary
        # If stopwords have to be removed, remove them
        if clean_stopwords:
            # Get the stopwords
            stopwords = nltk.corpus.stopwords.words(language)
            # Get the words that are not stopwords
            top_n_words = [word for word in top_n_words if word not in stopwords]

        top_n_words = top_n_words[:top_n]
        # Get the words
        highligth_words = [word[0] for word in top_n_words]
    else:
        highligth_words = [word[0] for word in top_n_words]

    highligth_words=list(set(highligth_words))
    return highligth_words

def get_words_indexes(text,word_list,ngrams=False):
    """
    Given a text and a list of words, return a dictionary with the start and end index of each word
    :param text: text
    :param word_list: list of words
    :return: list with the start and end index of each word
    """
    words_index = []
    for word in word_list:
        # if ngrams:
        #     word=" ".join(word)
        #remove the white spaces from the word
        word=word.strip()
        # Find all the start and end index of the token
        find_all_word_occurrences(text, word)
        if not ngrams:
            words_index.extend(find_all_word_occurrences(text, word))
        else:
            words_index.extend(find_all_grams_occurrences(text, word))

    # if there are any (0,0) tuples remove them
    words_index = [x for x in words_index if x != (0, 0)]

    return words_index
# A method to get the n grams with the highest attribution for a given label
def get_n_grams_attributions(n_grams, attributions_dict):
    """
    Given a list of n-grams and a dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label
    and the number of words in the n-gram, return a dictionary with the n-grams and their attributions
    :param n_grams: list of n-grams
    :param attributions_dict: dictionary of list of tuples with the attributions of each word or token (word, attribution)
    :return: dictionary with the n-grams and their attributions
    """
    # Build a dictionary with the n-grams and their attributions
    n_grams_attributions = {}
    for n_gram in n_grams:
        # Get the words of the n-gram
        words = [word for word in n_gram]
        # Get the attributions of each word
        words_attributions = []
        for word in words:
            # Get the attributions of the word
            word_attributions = attributions_dict[word]
            words_attributions.append(word_attributions)
        # Calculate the attribution of the n-gram
        n_gram_attribution = sum(words_attributions) / len(words_attributions)
        # Add the n-gram and its attribution to the dictionary
        n_grams_attributions[n_gram] = n_gram_attribution

    return n_grams_attributions

def get_top_n_grams(token_list,attributions_dict, top_highlighted_words,n=4,return_weights=False):
    """
    Given a token_list, a dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label,
    the top highlighted words and the number of words in the n-gram, return the top n n-grams and their attributions
    :param token_list: list of tokens
    :param attributions_dict: dictionary of list of tuples with the attributions of each word or token (word, attribution) for each label
    :param top_highlighted_words: top highlighted words
    :param n: number of words in the n-gram
    :param return_weights: if True, return the n-grams with their weights
    """

    n_grams=list(ngrams(token_list, n))
    #Calculate the attribution of each n-gram
    n_grams_attributions=get_n_grams_attributions(n_grams, attributions_dict)

    #Get for each highlighted word related n-grams
    n_grams_dict={}
    for word in top_highlighted_words:
        n_grams_list = []
        for n_gram in n_grams:
            if word in n_gram:
                n_grams_list.append(n_gram)
        n_grams_dict[word]=n_grams_list

    #Get the n-grams with the highest attribution
    top_n_grams={}
    for word in top_highlighted_words:
        #Get the n-grams related to the word
        n_grams_list=n_grams_dict[word]
        #Get the n-grams with the highest attribution
        top_n_grams[word]=sorted(n_grams_list, key=lambda x: n_grams_attributions[x], reverse=True)[:1]
    if return_weights:
        top_n_grams_grams= {}
        for word in top_n_grams:
            gram=top_n_grams[word][0]
            #turn the gram into a string
            gram_joined=" ".join(gram)
            top_n_grams_grams[gram_joined]=n_grams_attributions[gram]
        top_n_grams=top_n_grams_grams
    return top_n_grams

# def get_definition(icd_code,raw=False,language="english"):
#     """
#     Get definition of an ICD code
#     :param icd_code: ICD code
#     :param raw: if True, return raw definition else return clean definition
#     :return: definition
#     """
#     icd_code=icd_code.upper()
#     if language=="english":
#         definition=icd10_lookup_cm.get_description(icd_code)
#     else:
#         cie=CIECodes()
#         definition=cie.info(code=icd_code)
#         if definition is None:
#             definition=icd10_lookup_cm.get_description(icd_code)
#         else:
#             definition=definition["description"]
#     if not raw:
#         #Clean all special characters using regex
#         definition=re.sub(r'[^a-zA-Z0-9\s]', '', definition)
#
#         definition=definition.split(" ")
#         #Remove stop words
#         stopwords = nltk.corpus.stopwords.words(language)
#         definition = [word for word in definition if word not in stopwords]
#
#     return definition