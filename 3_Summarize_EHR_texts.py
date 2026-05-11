import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import argparse

#Imports for using langchain
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.schema.runnable import RunnablePassthrough

import time

#Imports for vector DB
import chromadb
from chromadb.utils import embedding_functions

#Imports for using vector DB with Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import os
from huggingface_hub import login
login(os.environ["HF_TOKEN"])



def get_explainable_prediction(model, tokenizer, ehr):
    prompt_template = """
   Eres un experto clínico. Tu tarea es escribir un resumen detallado sobre el registro de salud electrónico
    (EHR) de un paciente hospitalizado. 

    #Instrucciones:
        Estado Actual del Paciente:

        Redacta un resumen breve y preciso del estado actual del paciente en una sección titulada 
        "Enfermedad Actual".Describe la enfermedad o condición principal por la cual el paciente está 
        recibiendo atención en este momento, destacando los síntomas actuales, su intensidad, 
        duración, y cualquier cambio reciente en su condición.

        Historial Clínico:

        Revisa el historial médico del paciente y enumera sus enfermedades previas, condiciones crónicas o 
        cualquier otro antecedente relevante que pueda influir en su estado actual.
        Organiza esta información de manera cronológica o por importancia, y utiliza una lista ordenada 
        para que sea fácil de leer y entender.

        Resultados y Conclusiones de las Pruebas Clínicas:

        Analiza los resultados de las pruebas diagnósticas recientes (análisis de laboratorio, 
        estudios de imágenes, etc.). Por cada prueba menciona los 
        resultados númericos y extrae una conclusión escrita por cada uno de ellos.

        Impresión diagnóstica:

        Utiliza una lista para mencionar todos aquellos posibles diagnósticos de la enfermedad actual que se 
        mencionan en el EHR.

        No añadas más información que la solicitada.

   #Información:

        Este es el ehr del paciente:`{ehr}´
"""
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=1,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=1512,
    )
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["ehr"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    result = llm_chain.invoke({"ehr": ehr}, return_only_outputs=True)

    return result

def load_quantized_model(model_id):
    """
    Load a quantized model
    :param model_path: path to the model
    :return: model, tokenizer
    """
    # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # model_id="meta-llama/Llama-2-13b-chat-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_mistral = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)
    print("The model is ready")

    return model_mistral, eval_tokenizer

def load_mistral_model():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    #model_id="meta-llama/Llama-2-13b-chat-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_mistral = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)
    print("The model is ready")

    return model_mistral, eval_tokenizer


# generate main function
if __name__ == "__main__":

    #define argument parser
    parser = argparse.ArgumentParser(description="Summarize EHR texts")

    #path to the dataset
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--data_sep", type=str, default=",", help="Separator of the data")

    #name of the column containing the EHR text
    parser.add_argument("--ehr_column", type=str, help="Name of the column containing the EHR text")


    #model to generate the summary by default mixtral
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model to generate the summary")

    #path to save the resulting dataset with the summarized texts
    parser.add_argument("--output_path", type=str, help="Path to save the resulting dataset")

    #start and end lines
    parser.add_argument("--start_line", type=int, help="Start line to summarize", default=0)
    parser.add_argument("--end_line", type=int, help="End line to summarize",default=None)

    #parse the arguments
    args = parser.parse_args()


    print("Loading the model")
    # Load models
    model_mistral, eval_tokenizer = load_quantized_model(args.model)
    accelerator = Accelerator()
    model_mistral, eval_tokenizer = accelerator.prepare(model_mistral, eval_tokenizer)
    #load data
    data = pd.read_csv(args.data_path,sep=args.data_sep)

    #get the EHR texts
    ehr_texts = data[args.ehr_column].tolist()

    #get the texts summaries. Add a new column to the dataframe

    print("Summarizing the EHR texts")
    #add the summaries one by one and save the df to the output path once in 100 iterations
    data["summarized_text"] = "hola"
    start_line=args.start_line # Línea donde deseas comenzar
    if args.end_line is None:
        end_line=len(ehr_texts)
    else:
        end_line=args.end_line

    for i, ehr in enumerate(ehr_texts[start_line:end_line], start=start_line):

        # Temporizador para inicialización de Accelerator
        start_time = time.time()
        summary = get_explainable_prediction(model_mistral, eval_tokenizer, ehr)

        # data["summarized_text"][i] = summary

        print(f"Summary generated in {time.time() - start_time:.2f} seconds")

        data.at[i, "summarized_text"] = summary["text"]
        print(i)
        if i % 10 == 0:
            print(f"Saving data to {args.output_path}")
            print("Processed ", i, " EHR texts")
            data.to_csv(args.output_path)
    data.to_csv(args.output_path)





