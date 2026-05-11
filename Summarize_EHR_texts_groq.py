import argparse
import os
import time

import groq
import pandas as pd
from huggingface_hub import login
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

login(os.environ["HF_TOKEN"])



def get_explainable_prediction(model, tokenizer, ehr):
    chat = ChatGroq(temperature=0, groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="mixtral-8x7b-32768")
    prompt_template = """
   You are a clinical expert. Your task is to write a detailed summary of the electronic health record (EHR) for a hospitalized patient.

#Instructions:

Current Patient Status:

Write a brief and precise summary of the patient’s current status in a section titled "Present Illness."
Describe the main illness or condition for which the patient is receiving care at this time, highlighting current symptoms, their intensity, duration, and any recent changes in the condition.
Clinical History:

Review the patient’s medical history and list past illnesses, chronic conditions, or any other relevant background that could influence the current status.
Organize this information chronologically or by importance, and use a bulleted or numbered list for clarity and readability.
Test Results and Clinical Conclusions:

Analyze recent diagnostic test results (e.g., laboratory tests, imaging studies).
For each test, include numerical results and provide a written conclusion summarizing the findings.
Diagnostic Impressions:

Use a list to mention all potential diagnoses related to the current illness as referenced in the EHR.
Do not add any information beyond what is requested.

#Information:

This is the patient’s EHR: {ehr}
"""
    # text_generation_pipeline = transformers.pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     task="text-generation",
    #     temperature=1,
    #     repetition_penalty=1.1,
    #     return_full_text=False,
    #     max_new_tokens=1512,
    # )
    # mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt = ChatPromptTemplate.from_messages([("system", prompt_template)])
    llm_chain= prompt | chat

    # Create llm chain

    max_retries = 5  # Define un número máximo de intentos para evitar bucles infinitos
    retries = 0
    chunk_size = len(ehr)
    max_tokens = 5000  # Límite del modelo

    while retries < max_retries:
        try:
            # Invocar el modelo con el tamaño actual del dataset
            result = llm_chain.invoke({"ehr": ehr[:chunk_size]}, return_only_outputs=True)
            return result  # Devuelve el resultado si no hay errores

        except groq.APIStatusError as e:
            error_message = str(e)
            if "Request too large" in error_message:
                # Reduce dinámicamente el tamaño del EHR si ocurre el error
                chunk_size = int(chunk_size * 0.9)  # Reduce en un 10%
                retries += 1
                print(f"Reduciendo el tamaño del dataset. Nuevo tamaño: {chunk_size} tokens.")
            else:
                # Lanza el error si no es relacionado con el tamaño
                raise e

    # result = llm_chain.invoke({"ehr": ehr}, return_only_outputs=True)


    return result

# def load_quantized_model(model_id):
#     """
#     Load a quantized model
#     :param model_path: path to the model
#     :return: model, tokenizer
#     """
#     # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     # model_id="meta-llama/Llama-2-13b-chat-hf"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     model_mistral = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#
#     eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)
#     print("The model is ready")
#
#     return model_mistral, eval_tokenizer
#
# def load_mistral_model():
#     model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     #model_id="meta-llama/Llama-2-13b-chat-hf"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     model_mistral = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#
#     eval_tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True, trust_remote_code=True)
#     print("The model is ready")
#
#     return model_mistral, eval_tokenizer


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
    # model_mistral, eval_tokenizer = load_quantized_model(args.model)
    # accelerator = Accelerator()
    # model_mistral, eval_tokenizer = accelerator.prepare(model_mistral, eval_tokenizer)
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
        ehr=(" ".join(ehr.split()[:2000]))
        summary = get_explainable_prediction(None, None, ehr)

        #wait a minute due to rate limiting
        time.sleep(60)

        # print(summary.content)
        # data["summarized_text"][i] = summary

        print(f"Summary generated in {time.time() - start_time:.2f} seconds")

        data.at[i, "summarized_text"] = summary.content
        print(i)
        if i % 5 == 0:
            print(f"Saving data to {args.output_path}")
            print("Processed ", i, " EHR texts")
            data.to_csv(args.output_path)
       #For testing purposes

    data.to_csv(args.output_path)





