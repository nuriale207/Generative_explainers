# import sys
# sys.path.append("doxpy")
import argparse
import json
import re

import numpy as np
import pandas as pd
import simple_icd_10_cm as cm
from doxpy.misc.jsonld_lib import *
from doxpy.models.estimation.dox_estimator import DoXEstimator
from doxpy.models.knowledge_extraction.knowledge_graph_extractor import KnowledgeGraphExtractor
from doxpy.models.knowledge_extraction.knowledge_graph_manager import KnowledgeGraphManager
from doxpy.models.reasoning.answer_retriever import AnswerRetriever
from itertools import islice

# Define all default configurations

answer_pertinence_threshold=0.5

################ Configuration ################
answer_pertinence_threshold=0.5

################ Configuration ################
ARCHETYPE_FITNESS_OPTIONS = {
    'one_answer_per_sentence': False,
    'answer_pertinence_threshold': answer_pertinence_threshold,
    'answer_to_question_max_similarity_threshold': None,
    'answer_to_answer_max_similarity_threshold': 0.85,
}

KG_MANAGER_OPTIONS = {
    'spacy_model': 'es_dep_news_trf',
    'n_threads': 1,
    # 'use_cuda': True,
    'with_cache': False,
    'with_tqdm': False,

    # 'min_triplet_len': 0,
    # 'max_triplet_len': float('inf'),
    # 'min_sentence_len': 0,
    # 'max_sentence_len': float('inf'),
    # 'min_paragraph_len': 0,
    # 'max_paragraph_len': 0, # do not use paragraphs for computing DoX
}

KG_BUILDER_DEFAULT_OPTIONS = {
    'spacy_model': 'es_dep_news_trf',
    'n_threads': 1,
    # 'use_cuda': True,

    'with_cache': False,
    'with_tqdm': False,

    'max_syntagma_length': None,
    'add_source': True,
    'add_label': True,
    'lemmatize_label': False,

    # 'default_similarity_threshold': 0.75,
    'default_similarity_threshold': 0,
    'tf_model': {
        'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3',  # Transformer
        # 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
        # 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
        # 'use_cuda': True,
        # 'with_cache': True,
        # 'batch_size': 100,
    },
}

CONCEPT_CLASSIFIER_DEFAULT_OPTIONS = {
    'spacy_model': 'es_dep_news_trf',
    'n_threads': 1,
    # 'use_cuda': True,

    'default_batch_size': 20,
    'with_tqdm': False,

    'tf_model': {
        'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
        # 'url': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3',  # Transformer
        # 'url': 'https://tfhub.dev/google/universal-sentence-encoder/4', # DAN
        # 'cache_dir': '/Users/toor/Documents/Software/DLModels/tf_cache_dir/',
    },
    # 'sbert_model': {
    # 	'url': 'all-MiniLM-L12-v2',
    # 	'use_cuda': True,
    # },
    'default_similarity_threshold': 0.5,
    # 'with_stemmed_tfidf': True,
    'default_tfidf_importance': 0,
}

SENTENCE_CLASSIFIER_DEFAULT_OPTIONS = {
    'spacy_model': 'es_dep_news_trf',
    'n_threads': 1,
    # 'use_cuda': True,

    'sbert_model': {
        'url': 'multi-qa-MiniLM-L6-cos-v1',
        # 'use_cuda': True,
        'with_cache': True,
    },

    # 'default_batch_size': 100,
    'with_tqdm': False,
    'with_cache': False,

    'default_tfidf_importance': 0,
}


def get_questions_template(language:"english"):
    if language=="spanish":
        question_template_list=[
            'Que es {X}?',
            'Como se relaciona la historia del paciente con {X}?',
            'Como se relacionan los sintomas del paciente con {X}?',
            "Como se relacionan los medicamentos anteriores del paciente con {X}?",
            "Como se relacionan los procedimientos clinicos anteriores del paciente con {X}?",
            'Por que es plausible que el paciente sea diagnosticado con {X}?',
        ]
        question_template_weight_dict={
            "Que es {X}?": 0.2,
            "Como se relaciona la historia del paciente con {X}?": 0.5,
            "Como se relacionan los sintomas del paciente con {X}?": 0.35,
            "Como se relacionan los medicamentos anteriores del paciente con {X}?": 0.35,
            "Como se relacionan los procedimientos clinicos anteriores del paciente con {X}?": 0.35,
            "Por que es plausible que el paciente sea diagnosticado con {X}?": 0.45,
        }
    else:
        question_template_list = [ # Q: the archetypal questions
                'What is {X}?',
                'How is the patient history related to {X}?',
                'How are the patient symptoms related to {X}?',
                "How are the patient past medications related to {X}?",
                'How are the patient past procedures related to {X}?',
                'Why is it plausible for the patient to be diagnosed with {X}?',
        ]

        question_template_weight_dict = {
                "What is {X}?": 0.2,
                "How is the patient history related to {X}?": 0.5,
                "How are the patient symptoms related to {X}?": 0.35,
                "How are the patient past medications related to {X}?": 0.35,
                "How are the patient past procedures related to {X}?": 0.35,
                "Why is it plausible for the patient to be diagnosed with {X}?": 0.45,
            }
    return question_template_list,question_template_weight_dict

def get_explanation_dictionary(explanation):
    texto = explanation
    patron = r'### Explanation for the prediction:(i\d+)\s+([^\n]+)\n\n\s+(.+?)(?=\n###|$)'

    # Usamos re.findall para capturar todos los grupos de etiquetas y explicaciones
    matches = re.findall(patron, texto, re.DOTALL)

    # Creamos el diccionario

    explicaciones = {etiqueta: f'{titulo}\n\n{explicacion.strip()}' for etiqueta, titulo, explicacion in matches}

    #Remove all the text before "\n\n" in the explanation
    for key in explicaciones.keys():
        explicaciones[key] = explicaciones[key].split("\n\n")[1]

    return explicaciones


def get_explanandum_aspect_list(label,codes_dict,categ_dict):
    codes=expand_range(label.split("-")[0],label.split("-")[1])
    all_explanandum_labels=[label.upper()]
    all_explanandum_aspects=["my:"+categ_dict[label.upper()]]
    for code in codes:
        definition=codes_dict[code.upper()]
        if definition is not None and not (isinstance(definition, float) and math.isnan(definition)):
            all_explanandum_labels.append(code.upper())
            print(definition)
            all_explanandum_aspects.append("my:"+definition)

    return all_explanandum_labels, all_explanandum_aspects

def expand_range(start, end):
    start_prefix, start_num = start[:1], int(start[1:])
    end_prefix, end_num = end[:1], int(end[1:])
    codes = []
    for num in range(start_num, end_num + 1):
        codes.append(f"{start_prefix}{num:02d}")
    return codes

def get_doxpy_degree(PHI, EXPLANDUM_ASPECTS_LIST, answer_pertinence_threshold=0.5,language="english"):
    avg_doxpy = []
    for explanandum_aspect in EXPLANDUM_ASPECTS_LIST:

        print(explanandum_aspect)

        EXPLANANDUM_ASPECTS = [explanandum_aspect]

        question_template_list, question_template_weight_dict = get_questions_template(language)

        explainable_information_graph = KnowledgeGraphExtractor(KG_BUILDER_DEFAULT_OPTIONS).set_content_list(PHI,
                                                                                                             remove_stopwords=False,
                                                                                                             remove_numbers=False,
                                                                                                             avoid_jumps=True).build()

        kg_manager = KnowledgeGraphManager(KG_MANAGER_OPTIONS, explainable_information_graph)

        qa = AnswerRetriever(
            # Using qa_dict_list also for getting the archetype_fitness_dict might over-estimate the median pertinence of some archetypes (and in a different way for each), because the QA Extractor is set to prefer a higher recall to a higher precision.
            kg_manager=kg_manager,
            concept_classifier_options=CONCEPT_CLASSIFIER_DEFAULT_OPTIONS,
            sentence_classifier_options=SENTENCE_CLASSIFIER_DEFAULT_OPTIONS,
            # betweenness_centrality= betweenness_centrality,
        )

        ### Get explanandum aspects
        explanandum_aspect_list = EXPLANANDUM_ASPECTS
        print('Important explicandum aspects:', len(explanandum_aspect_list))
        print(json.dumps(explanandum_aspect_list, indent=4))

        ### Define a question generator
        question_generator = lambda question_template, concept_label: question_template.replace('{X}', concept_label)

        ### Initialise the DoX estimator
        dox_estimator = DoXEstimator(qa)
        ### Estimate DoX
        dox = dox_estimator.estimate(
            aspect_uri_iter=list(explanandum_aspect_list),
            query_template_list=question_template_list,
            question_generator=question_generator,
            **ARCHETYPE_FITNESS_OPTIONS,
        )
        print(f'DoX:', json.dumps(dox, indent=4))

        # Normalize the DoX values
        max_dox = max(dox.values())

        # avoid division by zero if max_dox is zero
        if max_dox == 0:
            max_dox = 1
        normalized_dox = {aspect: value / max_dox for aspect, value in dox.items()}

        ### Compute the average DoX

        weighted_averaged_dox = dox_estimator.get_weighted_degree_of_explainability(dox,
                                                                                    archetype_weight_dict=question_template_weight_dict)
        #         if weighted_averaged_dox>1:
        #             weighted_averaged_dox=dox_estimator.get_weighted_degree_of_explainability(normalized_dox, archetype_weight_dict=question_template_weight_dict)

        weighted_averaged_dox = weighted_averaged_dox / sum(question_template_weight_dict.values())
        ## Print all different DoX values
        print("Weighted Average DoX:", weighted_averaged_dox)

        avg_doxpy.append(weighted_averaged_dox)
    return avg_doxpy


#Define main function

if __name__== "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Evaluate explainability' )

    #Data arguments
    parser.add_argument('--data', type=str, help='Path to the data with the explanations to evaluate', default=None)
    # add argument to know the column with the explanations
    parser.add_argument('--explanation_column', type=str, help='Name of the column containing the explanations', default=None)
    #add argument to know the column with the label
    parser.add_argument('--label_column', type=str, help='Name of the column containing the labels', default=None)
    #language of the data
    parser.add_argument("--language",type=str,help="Language of the text model and the data: english or spanish. "
                                                   "Default english",default="english")

    #Save arguments
    parser.add_argument('--output', type=str, help='Path to the output file', default=None)

    args = parser.parse_args()

    # Load the data
    csv_file = pd.read_csv(args.data)

    cat2def = pd.read_csv("/scratch/nlebena/Datasets/MIMIC-IV/ICD-10/Categories/ICD-10_categories.csv")
    main2def = pd.read_csv("/scratch/nlebena/Datasets/MIMIC-IV/ICD-10/Categories/ICD-10_main_codes.csv")

    if args.language == "spanish":

        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definition_Spanish"]))
        main2def_dict = dict(zip(main2def["code"], main2def["definition_spanish"]))
    else:
        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definiton"]))
        main2def_dict = dict(zip(main2def["code"], main2def["definition"]))

    #Load the questions template
    question_template_list, question_template_weight_dict = get_questions_template(args.language)

    # save the json file
    output = args.output

    # Iterate over the rows of  resultado_dict column of the dataframe
    # Generate a json file with the DoX values

    csv_file["Dox"] = None

    for idx, row in csv_file[:300].iterrows():
        # for each explanation in the dictionary, get the DoX value
        PHI = row[args.explanation_column]
        label=row[args.label_column]
        # #     print(PHI)
        # id_ = row.HADM_ID
        # subject_id = row.Subject_ID

        #         print(key)
        _, EXPLANANDUM_ASPECTS_LIST = get_explanandum_aspect_list(label, main2def_dict, cat2def_dict)

        #         print(txt)

        try:
            # Llama a la función que podría generar el error
            avg_doxpy = get_doxpy_degree([PHI], EXPLANANDUM_ASPECTS_LIST, answer_pertinence_threshold=0.5,
                                         language=args.language)
            # Código adicional si la ejecución es exitosa
            # print("Calculo exitoso de avg_doxpy:", avg_doxpy)

        except ValueError as ve:
            # Manejo específico del ValueError
            print(f"Error de valor: {ve}")
            avg_doxpy = [0.0]

        except Exception as e:
            # Manejo genérico de otros posibles errores
            # print(f"Se encontró un error inesperado: {e}")
            print(f"Error: {ve}")
            avg_doxpy = [0.0]

        #add doxpy to the csv file
        csv_file.loc[idx, 'DoX'] = max(avg_doxpy)

        print("Proccessed up to row: ", idx)
        # save files when processed 200 documents
        if idx % 200 == 0:
            print("Saving files")
            csv_file.to_csv(args.output,index=False)


    csv_file.to_csv(args.output,index=False)