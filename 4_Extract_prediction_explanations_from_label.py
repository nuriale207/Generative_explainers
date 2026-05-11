#!/usr/bin/env python
# coding: utf-8
import argparse

from explainability_utils.explainability_utils import get_prediction_explanations

from utils.model_loader import load_quantified_model, get_prediction_labels
# In[10]:


from utils.website_functions_v2 import *
import pandas as pd
import warnings
from utils import model_handler, model_loader
import simple_icd_10_cm as cm

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
    parser.add_argument('--data_path', type=str, help='Path to the data with the data', default=None)
    parser.add_argument('--label_column', type=str, help='Name of the column with the label', default=None)
    parser.add_argument('--text_column', type=str, help='Name of the column with the text to explain', default=None)

    # Model arguments
    parser.add_argument('--generative_explainer', type=str, help='Name of the generative model to explain the output',
                        default=None)

    # Save arguments
    parser.add_argument('--output', type=str, help='Path to the output file', default=None)

    # Language argument
    parser.add_argument('--language', type=str, help='Language of the text: english or spanish', default='english')

    args = parser.parse_args()

    # Load models
    print("Loading model....")
    explainer_model, explainer_tokenizer = load_quantified_model(args.generative_explainer)

    # #### Main iteration

    # Get the cat2def dictionary
    cat2def = pd.read_csv("/home/nlebena/scratch/Datasets/MIMIC-IV/ICD-10/Categories/ICD-10_categories.csv")
    if args.language == "english":
        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definition"]))
    else:
        cat2def_dict = dict(zip(cat2def["Code-range"], cat2def["Definition_Spanish"]))

    print("Extracting explainable predictions....")

    contador = 0
    warnings.filterwarnings('ignore')
    # Crear un nuevo DataFrame para guardar los resultados. Tendrá las siguientes columnas:
    # - Subject_id -HADM_ID -Text -Label -Explanation

    # Creamos el df para ir llenandolo

    explanations_list = []
    # pd.DataFrame(columns=["Subject_id","HADM_ID","FILTERED_TEXT_CODES","Label","Explanation"])
    # Iterar sobre cada índice y fila del DataFrame
    for idx, fila in predictions.iterrows():

        label=fila[args.label_column]
        # get user ID
        user_id = fila["Subject_id"]
        hadm_id = fila["HADM_ID"]

        # Get t
        # get the text of the patient
        labels = get_prediction_labels(prediction_list, id2label)

        text, explanations = get_prediction_explanations(cat2def_dict, labels, fila["summarized_text_codes"],
                                                         explainer_model, explainer_tokenizer,
                                                         language=args.language, model_name=args.generative_explainer)

        # dividir el text en labels
        text = text.split("\n")

        # Por cada i en las labels añadir la explanation

        for i in range(len(labels)):
            # Add a new line to the pandas dataframe
            explanation_dict = {"Subject_id": user_id, "HADM_ID": hadm_id,
                                "summarized_text_codes": fila["summarized_text_codes"], "Label": labels[i],
                                "Explanation": explanations[i]}

            explanations_list.append(explanation_dict)

        # Guardar el DataFrame en un archivo CSV, con un nombre único para cada iteración

        # # Opcional: mostrar el DataFrame en cada paso
        # print(f"Iteración {idx}:")
        # print(predictions)

        # Incrementar el contador
        contador += 1

        # Imprimir mensaje cada 5 documentos y terminar la ejecución (para probar el software)
        if contador % 5 == 0:
            print(f"Processed up to {contador} documents.")
            explanations_df = pd.DataFrame(explanations_list)
            explanations_df.to_csv(args.output, index=False)

    # #### Load data

    print("Loading data....")

