#!/usr/bin/env python
# coding: utf-8
import argparse
import warnings

import pandas as pd

from explainability_utils.explainability_utils import get_prediction_explanations, get_explainable_prediction

from utils.model_loader import load_quantified_model, get_prediction_labels
from utils.website_functions_v2 import *

# In[10]:

# ### Load models

# In[7]:


#
# def load_model(sequence_model_path,text_model_path):
#
#     sequence_model,sequence_tokenizer,sequence_config=model_handler.load_model(sequence_model_path)
#
#     text_model,text_tokenizer,text_config=model_handler.load_model(text_model_path)
#
#
#     return sequence_model, sequence_tokenizer, text_model,text_tokenizer
#


# def load_chroma_db():
#     class MegatronEmbeddings(EmbeddingFunction):
#         def __call__(self, input: Documents) -> Embeddings:
#             tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
#             config = AutoConfig.from_pretrained(
#                 "/scratch/nlebena/Timeline_experiments/Experiments_data/MIMIC-IV_cardiology/"
#                 "MIMIC-IV_pretrained_gatortron_next_code/model/checkpoint-3000")
#             model = AutoModel.from_pretrained(
#                 "/scratch/nlebena/Timeline_experiments/Experiments_data/MIMIC-IV_cardiology/"
#                 "MIMIC-IV_pretrained_gatortron_next_code/model/checkpoint-3000")
#             model.to("cuda")
#             token_ids = tokenizer(input, return_tensors='pt')['input_ids']
#             token_ids = token_ids.to("cuda")
#             with torch.no_grad():
#                 outputs = model(token_ids)
#                 embeddings = outputs.last_hidden_state.mean(dim=1)  # Taking mean over all tokens
#                 embeddings = embeddings.cpu().squeeze().numpy()
#                 embeddings = embeddings.tolist()
#             return embeddings
#
#     chroma_client = chromadb.Client()
#     chroma_client = chromadb.PersistentClient(path="/scratch/nlebena/Datasets/MIMIC-III/Vector_database")
#     collection = chroma_client.get_or_create_collection(name="Patients_embeddings_fold_5",
#                                                         embedding_function=MegatronEmbeddings())
#     return chroma_client,collection


# generate main function

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Get explanations for the predictions of a model')

    # Data arguments
    parser.add_argument('--data_path', type=str, help='Path to the data with the ground truth labels', default=None)
    # add argument to know the column with the explanations
    parser.add_argument('--text_column', type=str, help='Name of the column containing the text',
                        default=None)
    # add argument to know the column with the label
    parser.add_argument('--label_column', type=str, help='Name of the column containing the labels', default=None)

    parser.add_argument('--generative_explainer', type=str, help='Name of the generative model to explain the output',
                        default=None)
    parser.add_argument("--language", type=str, help="Language of the text model and the data: english or spanish. "
                                                     "Default english", default="english")
    # Save arguments
    parser.add_argument('--output', type=str, help='Path to the output file', default=None)

    args = parser.parse_args()

    # Load models
    print("Loading model....")

    explainer_model, explainer_tokenizer = load_quantified_model(args.generative_explainer)

    # #### Main iteration
    predictions=pd.read_csv(args.data_path)

    # Get the cat2def dictionary
    cat2def = pd.read_csv("/home/nlebena/scratch/Datasets/MIMIC-IV/ICD-10/Categories/ICD-10_categories.csv")
    if args.language == "english":
        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definiton"]))
    else:
        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definition_Spanish"]))

    print("Extracting explainable predictions....")

    contador = 0
    warnings.filterwarnings('ignore')
    # Crear un nuevo DataFrame para guardar los resultados. Tendrá las siguientes columnas:
    # - Subject_id -HADM_ID -Text -Label -Explanation

    # Creamos el df para ir llenandolo

    predictions["New_explanations"]=""
    # pd.DataFrame(columns=["Subject_id","HADM_ID","FILTERED_TEXT_CODES","Label","Explanation"])
    # Iterar sobre cada índice y fila del DataFrame
    for idx, fila in predictions.iterrows():

        label=fila[args.label_column]
        text=fila[args.text_column]



        # Get t
        # get the text of the patient
        expl_pred = get_explainable_prediction(explainer_model, explainer_tokenizer, text, label, cat2def_dict[label.upper()],
                                               language=args.language, model_name=args.generative_explainer)
        predictions.loc[idx, "New_explanations"] = expl_pred["text"]
        # Guardar el DataFrame en un archivo CSV, con un nombre único para cada iteración
        print(expl_pred["text"])
        # # Opcional: mostrar el DataFrame en cada paso
        # print(f"Iteración {idx}:")
        # print(predictions)

        # Incrementar el contador
        contador += 1

        # Imprimir mensaje cada 5 documentos y terminar la ejecución (para probar el software)
        if contador % 5 == 0:
            print(f"Processed up to {contador} documents.")
            predictions.to_csv(args.output, index=False)
    predictions.to_csv(args.output, index=False)



