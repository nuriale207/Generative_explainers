import sys
sys.path.append('..')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
# Try to import get_definition, but it may not be available
try:
    from explainability_utils.word_analysis_utils import get_definition
except ImportError:
    get_definition = None
# from statistics import max
import tensorflow_hub as hub
import tensorflow as tf

import numpy as np
import seaborn as sns

#Load USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# module_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
model_USE = hub.load(module_url)
#
# module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
# load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
# embed = hub.load(module_url, options=load_options)



def get_understandability_degree(top_label_words,model="USE",save_maps=False,maps_path=None):
    """
    Compute understandability degree from top label words
    :param top_label_words: dictionary of top label words
    :return: understandability degree
    """
    understandability=[]
    for label in top_label_words:
        #Get label definition
        label_definition=get_definition(label)
        #Get understandability degree
        understandability_degree=get_understandability_degree_by_words(label_definition,top_label_words[label],model,
                                                                       return_matrix=True)
        understandability.append(np.max(understandability_degree))



    return max(understandability)


def get_docs_understandability_degree(top_label_words_dict,model="USE"):
    undertandability_degree=[]
    for dictionary in top_label_words_dict:
        only_words_dict = {key: [item[0] for item in value] for key, value in dictionary.items()}

        understandability=get_understandability_degree(only_words_dict,model="USE")
        undertandability_degree.append(understandability)
    return max(undertandability_degree)

def get_docs_understandability_degree_report(top_label_words_dict,model="USE",save_maps=False,maps_path=None):
    undertandability_degree = []
    for dictionary in top_label_words_dict:
        if n_grams:
            only_words_dict = {key: [" ".join(item[0]) for item in value] for key, value in dictionary.items()}
        else:
            only_words_dict = {key: [item[0] for item in value] for key, value in dictionary.items()}

        understandability = get_understandability_degree(only_words_dict, model="USE",save_maps=save_maps,maps_path=maps_path)
        undertandability_degree.append(understandability)

    overall_understandability = max(undertandability_degree)
    docs_understandability=undertandability_degree
    #Generate report
    report = "Overall understandability degree: " + str(overall_understandability) + "\n"
    for i in range(len(docs_understandability)):
        report += "Document " + str(i) + " understandability degree: " + str(docs_understandability[i]) + "\n"

    return report

def get_understandability_degree_by_words(def_words, inferred_words,model="USE",return_matrix=False):
    """
    Compute cosine similarity between two sets of words for understandability degree
    :param def_words: list of words
    :param inferred_words: list of words
    :return: cosine similarity
    """
    # def_embeddings=model_USE(def_words)

    # inferred_embeddings=model_USE(inferred_words)

    # similarity=cosine_similarity(inferred_embeddings,def_embeddings)

    # similarity_max=np.mean(similarity)


    # return similarity,similarity_max
    # Embeddings
    def_embeddings = model_USE(def_words)
    inferred_embeddings = model_USE(inferred_words)

    # Similarity matrix: shape (len(inferred_words), len(def_words))
    similarity = cosine_similarity(inferred_embeddings, def_embeddings)

    # Queremos: para cada palabra de la definición (columna), su similitud máxima
    # de entre todas las palabras inferidas.
    max_per_def_word = similarity.max(axis=0)   # axis=0 → columnas

    # Similaridad final = media de estos máximos
    similarity_score = max_per_def_word.mean()

    if return_matrix:
        return similarity, similarity_score

    return similarity_score

    # if not return_matrix:
    #     # Step 1: Find the maximum value in each row
    #     # max_values_by_row = np.max(similarity, axis=1)
    #
    #     # Step 2: Compute the mean of the maximum values
    #     # similarity_max = np.mean(max_values_by_row)
    #     similarity_max=np.mean(similarity)
    #     return similarity_max
    # else:
    #     return similarity



def get_by_label_docs_explainability_degree(top_word_weights,save_maps=False,maps_path=None,language="english",code_definitions_dict=None):
    """
    Compute explainability degree by label
    :param top_word_weights: dictionary of top word weights
    :param save_maps: whether to save by label heatmaps or not
    :param maps_path: path to save heatmaps
    :param language: language for definitions
    :param code_definitions_dict: optional dictionary mapping codes to definitions (code[:3].upper() -> definition)
    :return: a dictionary of the explainability degree by document
    """

    explainability_degree={}
    for label in top_word_weights:
        # Try to get definition from provided dictionary first, then fall back to get_definition
        if code_definitions_dict is not None:
            code_key = label[:3].upper() if len(label) >= 3 else label.upper()
            def_words = code_definitions_dict.get(code_key, None)
            if def_words is None:
                # Try with full label
                def_words = code_definitions_dict.get(label.upper(), None)
        else:
            def_words = None
        
        # If not found in dictionary, try get_definition function
        if def_words is None and get_definition is not None:
            try:
                def_words = get_definition(label, language=language)
            except:
                def_words = None
        
        # If still no definition found, skip this label
        if def_words is None:
            continue
        
        # Ensure def_words is a string (split into words if needed)
        if isinstance(def_words, str):
            # Split definition into words for similarity calculation
            import re
            def_words = re.sub(r'[^a-zA-Z0-9\s]', '', def_words).split()
        
        words=list(top_word_weights[label].keys())
        if not save_maps:
            explainability=get_understandability_degree_by_words(def_words,words,model="USE",return_matrix=True)
            if isinstance(explainability, tuple):
                similarity_matrix, similarity_score = explainability
                # Usar el similarity_score que ya tiene el cálculo correcto
                # (máximo por palabra de definición, luego promedio)
                explainability_degree[label] = similarity_score
            else:
                # Si no es tupla, es el score directamente
                explainability_degree[label] = explainability
        else:
            explainability = get_understandability_degree_by_words(def_words, words, model="USE",return_matrix=True)
            if isinstance(explainability, tuple):
                similarity_matrix, similarity_score = explainability
                # Usar el similarity_score que ya tiene el cálculo correcto
                explainability_degree[label] = similarity_score
                # Guardar la matriz para el heatmap
                explainability = similarity_matrix
            else:
                # Si no es tupla, no tenemos la matriz, no podemos generar el heatmap
                explainability_degree[label] = explainability
                continue
            #Save maps
            generate_explainability_degree_heatmap(explainability,def_words,words,maps_path+"/"+label+".png")
    
    doc_explainability_degree=np.mean(list(explainability_degree.values())) if explainability_degree else 0.0
    explainability_degree={"overall":doc_explainability_degree,"by_label":explainability_degree}
    return explainability_degree


def generate_explainability_degree_heatmap(sim_matrix,def_words,words, maps_path):
    # Create a heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(100, 100))
    sns.heatmap(sim_matrix, annot=True, cmap="YlGnBu", fmt=".2f", square=True, xticklabels=def_words,
                yticklabels=words)

    # Add axis labels
    plt.xlabel("Diagnostic term words")
    plt.ylabel("Definition words")

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # Save the heatmap
    # plt.show()
    plt.savefig(maps_path)


def generate_explainability_degree_heatmap_streamlit(sim_matrix, def_words, words, title="Explainability Degree Heatmap"):
    """
    Generate a heatmap for explainability degree that can be displayed in Streamlit
    :param sim_matrix: similarity matrix (numpy array)
    :param def_words: list of words from definition
    :param words: list of highlighted words
    :param title: title for the heatmap
    :return: matplotlib figure object
    """
    # Calculate appropriate figure size based on number of words
    num_def_words = len(def_words)
    num_highlighted_words = len(words)
    
    # Adjust size dynamically but keep it reasonable for Streamlit
    width = max(8, min(20, num_def_words * 0.8))
    height = max(6, min(15, num_highlighted_words * 0.6))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Create heatmap
    sns.heatmap(
        sim_matrix, 
        annot=True, 
        cmap="YlGnBu", 
        fmt=".2f", 
        square=False, 
        xticklabels=def_words,
        yticklabels=words,
        ax=ax,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    # Add axis labels
    ax.set_xlabel("Palabras de la definición", fontsize=12, fontweight='bold')
    ax.set_ylabel("Palabras destacadas", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig
