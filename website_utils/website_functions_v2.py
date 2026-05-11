import os
import re

import math
from collections import defaultdict
import groq
import time
import chromadb
from threading import Lock
import torch
from bs4 import BeautifulSoup
import pandas as pd
from datasets import Dataset
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig,
                          AutoModel, AutoModelForSequenceClassification)
# Recursos compartidos para Mixtral HF
_mixtral_model = None
_mixtral_tokenizer = None
_mixtral_lock = Lock()


def _get_or_load_mixtral_model(model_id: str):
    """Carga el modelo/tokenizer sólo una vez y los reutiliza."""
    global _mixtral_model, _mixtral_tokenizer

    if _mixtral_model is not None and _mixtral_tokenizer is not None:
        return _mixtral_model, _mixtral_tokenizer

    with _mixtral_lock:
        if _mixtral_model is not None and _mixtral_tokenizer is not None:
            return _mixtral_model, _mixtral_tokenizer

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception:
            # Aseguramos dejar el estado consistente si falla la carga
            _mixtral_model = None
            _mixtral_tokenizer = None
            raise

        _mixtral_model = model
        _mixtral_tokenizer = tokenizer
        return _mixtral_model, _mixtral_tokenizer


def unload_mixtral_resources():
    """Libera explícitamente los recursos del modelo Mixtral."""
    global _mixtral_model, _mixtral_tokenizer
    with _mixtral_lock:
        if _mixtral_model is not None:
            del _mixtral_model
            _mixtral_model = None
        if _mixtral_tokenizer is not None:
            del _mixtral_tokenizer
            _mixtral_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1667,  # ~10 req/min para no pasarnos
    check_every_n_seconds=0.1,
    max_bucket_size=30
)
#Imports for using langchain
# from langchain.schema.runnable import RunnablePassthroughs
def get_sequence_and_code_list(sex, age, patient_known_diagnoses, patient_known_diagnoses_text):
    codes = (" ").join(patient_known_diagnoses) + " " + patient_known_diagnoses_text
    codes = codes.replace(",", "")
    codes = codes.rstrip()
    codes = codes.lstrip()

    list_codes = codes.split(" ")
    list_codes=[code[:3] for code in list_codes if(len(code)>2)]
    print(list_codes)
    patient_input = sex + age + codes

    return patient_input, codes, list_codes


def get_patient_embedding(collection, sex, age, codes, patient_input):
    collection.add(
        documents=[patient_input],
        metadatas=[{"sex_text": sex, "age": age, "codes": codes}],
        ids=[str(patient_input)]
    )

    patient_embedding = collection.get(
        ids=[patient_input],
        include=["embeddings"],
    )

    collection.delete(
        ids=[patient_input]
    )

    patient_embedding = patient_embedding["embeddings"][0]

    return patient_embedding


def get_close_patient_embeddings(collection, patient_embedding, n=30):
    results = collection.query(
        query_embeddings=patient_embedding,
        n_results=n,
        include=["embeddings", "documents", "distances", "metadatas"],
    )

    return results


def get_same_diagnoses_patients(collection, code_list, sex, patient_age):
    if len(code_list) > 1:
        # Create the query structure
        document_query = {
            "$or": [{"$contains": string} for string in code_list]
        }
    else:
        document_query = {
            "$contains": code_list[0]
        }
    print(document_query)

    age_top = patient_age + 10
    age_min = patient_age - 10
    if sex is not None:
        metadata_query = {"$and": [{"sex": sex}, {"age": {"$lte": age_top}}, {"age": {"$gte": age_min}}]}
    else:
        metadata_query = {"$and": [{"age": {"$lte": age_top}}, {"age": {"$gte": age_min}}]}

    print(metadata_query)

    all_embeddings = collection.get(
        include=["embeddings", "documents", "metadatas"],
        where_document=document_query,
        where=metadata_query
    )
    return all_embeddings

# def get_prediction_text_model(text_model, text_tokenizer, patient_text, n=10):
#     # Access id2label and label2id
#     print("Getting prediction from text model")
#     id2label = text_model.config.id2label
#     print(id2label)
#
#     # Get text model prediction with truncation and padding
#     inputs_text = text_tokenizer(
#         patient_text,
#         return_tensors="pt",
#         truncation=True,   # Truncar el texto si excede la longitud máxima
#         padding="max_length",  # Rellenar para que todas las secuencias tengan la misma longitud
#         max_length=512  # Longitud máxima permitida por el modelo
#     )
#
#
#     outputs_text = text_model(**inputs_text)
#     predictions_text = torch.nn.functional.softmax(outputs_text.logits, dim=-1)
#
#     # Get top n predictions
#     topk_probabilities, topk_indices = torch.topk(predictions_text, n)
#
#     # Convert indices to labels
#     topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]
#
#     result = {}
#     result_text = ""
#     for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
#         result[label] = prob.item()
#         result_text += f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"
#         print(f"Top {i + 1}: {label} with probability {prob.item():.4f}")
#
#     return result, result_text
    
""" def get_prediction_text_model(text_model, text_tokenizer, patient_text, n=10,max_length=512):
    # Access id2label
    id2label = text_model.config.id2label

    # Tokenize on CPU by default
    inputs_text = text_tokenizer(
        patient_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # Now move everything in 'inputs_text' to GPU
    inputs_text = {k: v.to("cuda") for k, v in inputs_text.items()}

    # Model should already be on GPU: text_model.to("cuda")
    outputs_text = text_model(**inputs_text)

    predictions_text = torch.sigmoid(outputs_text.logits)

    # Get top n predictions
    topk_probabilities, topk_indices = torch.topk(predictions_text, n)

    # Convert indices to labels
    topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]

    result = {}
    result_text = ""
    for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
        result[label] = prob.item()
        result_text += f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"

    return result, result_text """

def get_prediction_text_model(text_model, text_tokenizer, patient_text, n=10, max_length=512):
    # Set model to eval mode
    text_model.eval()

    # Tokenize input
    inputs_text = text_tokenizer(
        patient_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # Move tokens to the same device as the model
    device = next(text_model.parameters()).device  # gpu or cpu
    inputs_text = {k: v.to(device) for k, v in inputs_text.items()}

    # Get logits without tracking gradients
    with torch.no_grad():
        outputs_text = text_model(**inputs_text)

    # Apply sigmoid
    probabilities = torch.sigmoid(outputs_text.logits)

    # Apply threshold 0.5 (same as in trainer)
    preds_binary = (probabilities > 0.5).float()

    # Get top n labels (like original)
    id2label = text_model.config.id2label
    topk_probabilities, topk_indices = torch.topk(probabilities, n)

    topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]

    # Format result
    result = {}
    result_text = ""
    for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
        result[label] = prob.item()
        result_text += f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"

    return result, result_text, preds_binary.cpu().tolist()


def get_prediction_classification_model(text_model, text_tokenizer, patient_text, n=10, max_length=512):
    """
    Obtiene predicciones del modelo replicando EXACTAMENTE el proceso de Train_transformer.py
    
    Args:
        text_model: Modelo de clasificación cargado
        text_tokenizer: Tokenizer usado para entrenar el modelo
        patient_text: Texto del paciente a predecir (string)
        n: Número de top labels a retornar
        max_length: Longitud máxima (debe coincidir con args.max_len usado en entrenamiento)
        use_exact_replication: Si True, usa el método exacto de DataHandler (recomendado)
    
    Returns:
        result: Diccionario con top n labels y probabilidades
        result_text: String formateado con las predicciones
        preds_binary: Lista binaria de predicciones (threshold 0.5)
    """
    text_model.eval()
    
    #Dataset temporal (como lo hace DataHandler)
    df_temp = pd.DataFrame({'text': [patient_text]})
    temp_dataset = Dataset.from_pandas(df_temp)
    
    # Replicar exactamente preprocess_data de DataHandler
    def preprocess_like_datahandler(examples):
        text = examples["text"]  # text es una lista cuando batched=True
        # Tokenizar EXACTAMENTE como en DataHandler línea 63
        encoding = text_tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        return encoding
    
    # Aplicar con batched=True (igual que DataHandler línea 77)
    tokenized_data = temp_dataset.map(preprocess_like_datahandler, batched=True)
    tokenized_data.set_format("torch")  # Igual que DataHandler línea 78
    
    # Obtener el primer elemento (formato que recibe el modelo)
    sample = tokenized_data[0]
    
    # Preparar inputs para el modelo
    inputs_text = {
        'input_ids': sample['input_ids'].unsqueeze(0),  # Añadir dimensión de batch
        'attention_mask': sample['attention_mask'].unsqueeze(0)
    }
    # Mover tokens al mismo dispositivo que el modelo
    device = next(text_model.parameters()).device
    inputs_text = {k: v.to(device) for k, v in inputs_text.items()}
    
    # Obtener logits sin tracking de gradientes
    with torch.no_grad():
        outputs_text = text_model(**inputs_text)
    
    # Aplicar sigmoid EXACTAMENTE como en Train_transformer.py (línea 220)
    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(outputs_text.logits)
    
    # Aplicar threshold 0.5 EXACTAMENTE como en Train_transformer.py (líneas 221-222)
    prediction_bin = probabilities.clone()
    prediction_bin[prediction_bin > 0.5] = 1
    prediction_bin[prediction_bin <= 0.5] = 0
    
    # Convertir a lista (igual que Train_transformer.py línea 223)
    preds_binary = prediction_bin[0].cpu().tolist()
    probabilities_list = probabilities[0].cpu().tolist()
    
    # Obtener top n labels
    id2label = text_model.config.id2label
    # Abrir y cargar el JSON como diccionario
    # with open("/home/nlebena/scratch/Classification_experiments/Experiments_data/osa_1370_mullenbach_clinical_longformer/test_v2/id2label.json", 'r') as f:
    #     content = f.read()
    # id2label = ast.literal_eval(content)

    topk_probabilities, topk_indices = torch.topk(probabilities, n)
    
    topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]
    
    # Formatear resultado
    result = {}
    result_text = ""
    
    for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
        result[label] = prob.item()
        result_text += f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"
    
    return result, result_text, preds_binary



def extraer_colores_por_codigo(html):
    soup = BeautifulSoup(html, "html.parser")

    codigo_a_color = {}

    spans = soup.find_all("span")

    for sp in spans:
        text = sp.get_text(strip=True)
        
        # Buscar códigos tipo i10, j4551, z9981, etc.
        if not re.fullmatch(r"[a-zA-Z]\d+", text):
            continue
        
        # Buscar el color en el span actual
        style = sp.get("style", "")
        m = re.search(r"color:\s*(#[0-9a-fA-F]{6})", style)
        
        # Si no está en el actual, mirar el padre
        if not m and sp.parent.name == "span":
            parent_style = sp.parent.get("style", "")
            m = re.search(r"color:\s*(#[0-9a-fA-F]{6})", parent_style)
        
        if m:
            codigo_a_color[text] = m.group(1)

    return codigo_a_color

def generate_expanding_summary(previous_summary, current_summary):
    chat = ChatGroq(temperature=0, groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="mixtral-8x7b-32768", rate_limiter=rate_limiter)
    prompt_template = """
        Eres un experto asistente clínico de IA encargado de COMBINAR DOS resúmenes de Historias Clínicas Electrónicas (HCE) DE UN MISMO PACIENTE en un solo resumen integral. Tu objetivo es crear un nuevo resumen que condense toda la información del paciente presente en ambos resúmenes de entrada, sin perder ningún detalle esencial.

        Se te proporcionarán dos resúmenes de HCE, cada uno con las siguientes secciones:
        - Estado Actual del Paciente
        - Historial Clínico
        - Resultados y Conclusiones de las Pruebas Clínicas
        - Impresión diagnóstica

        A continuación, se presentan los dos resúmenes de HCE:

        <resumen1>
        {resumen1}
        </resumen1>

        <resumen2>
        {resumen2}
        </resumen2>

        Tu tarea es crear un nuevo resumen combinado que mantenga las mismas cuatro secciones de los resúmenes de entrada. El resumen final debe  ser completo, informativo y no debe perder ninguna información esencial de los resúmenes originales.

        Guidelines for summary combination:

        1. Fusiona la información similar o idéntica presente en ambos resúmenes.
        2. Incluye la información única de cada resumen.
        3. IMPORTANTE: la sección de "Impresión diagnóstica" DEBE enumerar TODOS los diagnósticos de ambos resúmenes de origen (evitando duplicados).

        Presenta tu resumen combinado entre etiquetas <resumen3> utilizando la MISMA estructura de secciones que los resúmenes de origen, es decir: "Estado Actual del Paciente", "Historial Clínico", "Resultados y Conclusiones de las Pruebas Clínicas", "Impresión diagnóstica".

        Sigue ESTRICTAMENTE este formato en tu respuesta:
        <resumen3>
        {{contenidos resumen combinado estructurado utilizando la MISMA estructura de secciones}}
        </resumen3>

        Recuerda que es fundamental que tu resumen combinado conserve toda la información esencial de ambos resúmenes de entrada. Asegúrate de que no se pierdan detalles importantes durante el proceso de fusión. Utiliza todos los tokens que sean necesarios. JAMÁS intentes ahorrar tokens.

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

    if previous_summary != "":
        prompt = ChatPromptTemplate.from_messages([("system", prompt_template)])
        # Create llm chain
        llm_chain = prompt | chat

        max_retries = 500  # Define un número máximo de intentos para evitar bucles infinitos
        retries = 0
        retry_sleep = 75
        max_tokens = 5000  # Límite del modelo
        previous_summary_size = len(previous_summary)
        current_summary_size = len(current_summary)

        while retries < max_retries:
            try:
                result = llm_chain.invoke(
                    {"resumen1": previous_summary[:previous_summary_size],
                     "resumen2": current_summary[:current_summary_size]},
                    return_only_outputs=True
                )
                return result.content
            except groq.APIStatusError as e:
                if hasattr(e, "status_code") and e.status_code == 429:
                    print(f"[ERROR 429] Rate limited. Esperamos {retry_sleep}s y reintentamos...")
                    time.sleep(retry_sleep)
                    retries += 1
                elif hasattr(e, "status_code") and e.status_code == 413:
                    print(f"[ERROR 412] Request too large. Reduciendo tamaño de resúmenes...")
                    previous_summary_size = int(previous_summary_size * 0.9)  # Reduce en un 10%
                    current_summary_size = int(current_summary_size * 0.9)  # Reduce en un 10%
                else:
                    print(f"[ERROR] {e}")
                    raise e
            except Exception as e:
                print(f"[ERROR Genérico] {e}")
                break

        # Si llegamos aquí, falló tras max_retries
        print("Se alcanzó el número máximo de reintentos, devolviendo current_summary sin combinar.")
        return current_summary

    else:
        result = current_summary
    # result = llm_chain.invoke({"ehr": ehr}, return_only_outputs=True)

    return result

def get_mixtral_summary(ehr):
    chat = ChatGroq(temperature=0, groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="mixtral-8x22b", rate_limiter=rate_limiter)
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

    max_retries = 500  # Define un número máximo de intentos para evitar bucles infinitos
    retries = 0
    retry_sleep = 75
    max_tokens = 5000  # Límite del modelo
    chunk_size = len(ehr)

    while retries < max_retries:
        try:
            # Invocar el modelo con el tamaño actual del dataset
            result = llm_chain.invoke({"ehr": ehr[:chunk_size]}, return_only_outputs=True)
            return result  # Devuelve el resultado si no hay errores

        except groq.APIStatusError as e:

            if hasattr(e, "status_code") and e.status_code == 429:

                print(f"[ERROR 429] Rate limited. Esperamos {retry_sleep}s y reintentamos...")
                time.sleep(retry_sleep)
                retries += 1

            elif hasattr(e, "status_code") and e.status_code == 413:

                print(f"[ERROR 412] Request too large. Reduciendo tamaño de resúmenes...")
                chunk_size = int(chunk_size * 0.9)  # Reduce en un 10%
            else:
                print(f"[ERROR] {e}")
                raise e

        except Exception as e:
            print(f"[ERROR Genérico] {e}")
            break
    # Si llegamos aquí, falló tras max_retries
    print("Se alcanzó el número máximo de reintentos, devolviendo current_summary sin combinar.")
    return ""
    # result = llm_chain.invoke({"ehr": ehr}, return_only_outputs=True)



def get_mixtral_summary_hf(ehr: str, max_new_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Genera un resumen usando Hugging Face con el modelo
    mistralai/Mixtral-8x7B-Instruct-v0.1 y libera la GPU al finalizar.
    """
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    prompt_template = (
        "Eres un experto clínico. Tu tarea es escribir un resumen detallado sobre el registro de salud electrónico\n"
        "(EHR) de un paciente hospitalizado.\n\n"
        "#Instrucciones:\n"
        "Estado Actual del Paciente:\n"
        "Redacta un resumen breve y preciso del estado actual del paciente en una sección titulada \n"
        "\"Enfermedad Actual\". Describe la enfermedad o condición principal por la cual el paciente está \n"
        "recibiendo atención en este momento, destacando los síntomas actuales, su intensidad, duración, y cualquier cambio reciente.\n\n"
        "Historial Clínico:\n"
        "Revisa el historial médico del paciente y enumera sus enfermedades previas, condiciones crónicas o antecedentes relevantes.\n"
        "Organiza de forma cronológica o por importancia con listas donde aplique.\n\n"
        "Resultados y Conclusiones de las Pruebas Clínicas:\n"
        "Analiza las pruebas diagnósticas recientes (laboratorio, imagen, etc.), incluyendo valores numéricos y una conclusión por cada una.\n\n"
        "Impresión diagnóstica:\n"
        "Presenta una lista con todos los posibles diagnósticos mencionados en el EHR.\n\n"
        "No añadas información fuera de lo indicado.\n\n"
        "#Información:\n"
        f"Este es el ehr del paciente: `{ehr}`\n"
    )

    model, tokenizer = _get_or_load_mixtral_model(model_id)

    # Si el tokenizer tiene plantilla de chat, usarla; si no, usar prompt directo
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_template}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt_template

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # Guardar longitud del input

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=0.9
        )
    # Extraer solo los tokens generados (excluyendo el prompt de entrada)
    generated_tokens = outputs[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text

def generate_expanding_summary_hf(previous_summary: str, current_summary: str, max_new_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Genera un resumen expandido combinando dos resúmenes usando Hugging Face con el modelo
    mistralai/Mixtral-8x7B-Instruct-v0.1. Reutiliza el modelo cargado globalmente.
    """
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Si no hay resumen previo, devolver el actual
    if previous_summary == "":
        return current_summary
    
    prompt_template = (
        "Eres un experto asistente clínico de IA encargado de COMBINAR DOS resúmenes de Historias Clínicas Electrónicas (HCE) DE UN MISMO PACIENTE en un solo resumen integral. Tu objetivo es crear un nuevo resumen que condense toda la información del paciente presente en ambos resúmenes de entrada, sin perder ningún detalle esencial.\n\n"
        "Se te proporcionarán dos resúmenes de HCE, cada uno con las siguientes secciones:\n"
        "- Estado Actual del Paciente\n"
        "- Historial Clínico\n"
        "- Resultados y Conclusiones de las Pruebas Clínicas\n"
        "- Impresión diagnóstica\n\n"
        "A continuación, se presentan los dos resúmenes de HCE:\n\n"
        "<resumen1>\n"
        "{resumen1}\n"
        "</resumen1>\n\n"
        "<resumen2>\n"
        "{resumen2}\n"
        "</resumen2>\n\n"
        "Tu tarea es crear un nuevo resumen combinado que mantenga las mismas cuatro secciones de los resúmenes de entrada. El resumen final debe ser completo, informativo y no debe perder ninguna información esencial de los resúmenes originales.\n\n"
        "Guidelines for summary combination:\n\n"
        "1. Fusiona la información similar o idéntica presente en ambos resúmenes.\n"
        "2. Incluye la información única de cada resumen.\n"
        "3. IMPORTANTE: la sección de \"Impresión diagnóstica\" DEBE enumerar TODOS los diagnósticos de ambos resúmenes de origen (evitando duplicados).\n\n"
        "Presenta tu resumen combinado entre etiquetas <resumen3> utilizando la MISMA estructura de secciones que los resúmenes de origen, es decir: \"Estado Actual del Paciente\", \"Historial Clínico\", \"Resultados y Conclusiones de las Pruebas Clínicas\", \"Impresión diagnóstica\".\n\n"
        "Sigue ESTRICTAMENTE este formato en tu respuesta:\n"
        "<resumen3>\n"
        "{{contenidos resumen combinado estructurado utilizando la MISMA estructura de secciones}}\n"
        "</resumen3>\n\n"
        "Recuerda que es fundamental que tu resumen combinado conserve toda la información esencial de ambos resúmenes de entrada. Asegúrate de que no se pierdan detalles importantes durante el proceso de fusión. Utiliza todos los tokens que sean necesarios. JAMÁS intentes ahorrar tokens."
    )
    
    model, tokenizer = _get_or_load_mixtral_model(model_id)
    
    # Formatear el prompt con los resúmenes
    formatted_prompt = prompt_template.format(
        resumen1=previous_summary,
        resumen2=current_summary
    )
    
    # Si el tokenizer tiene plantilla de chat, usarla; si no, usar prompt directo
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": formatted_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = formatted_prompt
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # Guardar longitud del input
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Generación greedy (más rápida)
            temperature=temperature,
            top_p=0.9
        )
    # Extraer solo los tokens generados (excluyendo el prompt de entrada)
    generated_tokens = outputs[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text

def merge_summary_codes(summary_text,codes,code_def_dict):
    """
    Merge the summary text with the codes and their definitions
    :param summary_text:
    :param codes:
    :param code_def_dict:
    :return:
    """
    summary_text=summary_text+"\n"
    print(codes)
    for code in codes:
        print(code)
        # Usar .get() para evitar KeyError si el código no existe
        #code_def = code_def_dict.get(code)

        if code in code_def_dict:
            code_def = code_def_dict[code]
        else:
            code_def = None

        if code_def is not None and not (isinstance(code_def, float) and math.isnan(code_def)):
            summary_text=summary_text+code+": "+code_def+"\n"
        else:
            summary_text=summary_text+code+": "+"\n"

    return summary_text

def get_prediction_ensemble(sequence_model,sequence_tokenizer, patient_known_diagnoses,
                            text_model,text_tokenizer,patient_text,n=10):
    # Access id2label and label2id
    print("Getting prediction")
    id2label = sequence_model.config.id2label
    print(id2label)
    inputs = sequence_tokenizer(patient_known_diagnoses, return_tensors="pt")
    outputs = sequence_model(**inputs)
    predictions_sequence = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get text model prediction
    inputs_text = text_tokenizer(patient_text, return_tensors="pt")
    outputs_text = text_model(**inputs_text)
    predictions_text= torch.nn.functional.softmax(outputs_text.logits, dim=-1)


    #Get the weighted average ensemble of the two models
    weights=[0.7194,0.6798]
    predictions_ensemble=weights[0]*predictions_sequence+weights[1]*predictions_text

    topk_probabilities, topk_indices = torch.topk(predictions_ensemble, n)


    # Convert indices to labels
    topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]

    result = {}
    # Display the results
    result_text = ""
    for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
        result[label] = prob
        result_text = result_text + f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"
        print(f"Top {i + 1}: {label} with probability {prob.item():.4f}")

    return result, result_text

def get_prediction_inference(model,tokenizer,patient_known_diagnoses,n=10):
    # Access id2label and label2id
    print("Getting prediction")
    id2label = model.config.id2label
    print(id2label)

    inputs = tokenizer(patient_known_diagnoses, return_tensors="pt")
    outputs = model(**inputs)
    # Process the prediction to get top 10 classes
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    topk_probabilities, topk_indices = torch.topk(predictions, n)

    # Convert indices to labels
    topk_labels = [id2label[idx.item()] for idx in topk_indices[0]]

    result={}
    # Display the results
    result_text=""
    for i, (prob, label) in enumerate(zip(topk_probabilities[0], topk_labels)):
        result[label]=prob.item()
        result_text=result_text+f"Top {i + 1}: {label} with probability {prob.item():.4f}\n"
        print(f"Top {i + 1}: {label} with probability {prob.item():.4f}")

    return result, result_text


def get_prediction_masked(model,tokenizer,patient_known_diagnoses,n=10, specific_vocab=None):
    print("Getting prediction masked")

    patient_known_diagnoses=patient_known_diagnoses+" [MASK]"
    # Define your specific vocabulary list
    # print(specific_vocab[:100])
    specific_vocab_ids = {tokenizer.convert_tokens_to_ids(word) for word in specific_vocab}
    # print(specific_vocab_ids)

    inputs = tokenizer(patient_known_diagnoses, return_tensors="pt")

    # Generate outputs
    outputs = model(**inputs)
    # Get logits of the last token produced (what the model thinks comes next)
    logits = outputs.logits[:, -1, :]

    # If a specific vocabulary is provided, mask the logits for all other words
    if specific_vocab is not None:
        # Get logits of the masked token (assuming the masked token's logits are what you want)
        mask_index = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0]
        logits = outputs.logits[0, mask_index, :].squeeze(0)
        print("Logits")
        print(logits)
        # Apply mask to logits
        # Use a large negative number to mask out unwanted tokens
        masked_logits = logits.clone()
        for idx in range(logits.size(0)):
            if idx not in specific_vocab_ids:
                masked_logits[idx] = float('-inf')
        logits = masked_logits

    # Convert logits to probabilities (not necessary for top-k, but useful for inspecting)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    print(probabilities)
    # Get top n predictions
    top_probs, top_indices = torch.topk(probabilities, n)
    print("Top probabilities and indices")
    print(top_probs)
    print(top_indices)
    # Decode the predicted token IDs to tokens
    predicted_tokens = [tokenizer.decode(idx) for idx in top_indices]

    # Print the results
    result = {}
    # Display the results
    result_text = ""
    for i, (prob, token) in enumerate(zip(top_probs, predicted_tokens)):
        result[token] = prob
        result_text = result_text + f"Top {i + 1}: {token} with probability {prob.item():.4f}\n"
        print(f"Top {i + 1}: {token} with probability {prob.item():.4f}")

    return result, result_text


def count_code_occurrences(code_lists, codes_to_check):

    # Dictionary to hold counts of individual codes
    individual_counts = defaultdict(int)
    # Count of lists containing all codes
    all_codes_count = 0

    # Iterate over each list of codes
    for code_list in code_lists:
        # Convert the list to a set for fast lookup
        code_set = set(code_list)

        # Check for individual code occurrences
        for code in code_set:
            if code in codes_to_check:
                individual_counts[code] += 1

        # Check if all specified codes are present in the current list
        if all(code in code_set for code in codes_to_check):
            all_codes_count += 1

    return individual_counts, all_codes_count


def load_model(model_path, inference_model_path):


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    inference_model_LM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    inference_model_sequence = AutoModelForSequenceClassification.from_pretrained(
        inference_model_path,
        trust_remote_code=True
    )


    return model, tokenizer, inference_model_LM,inference_model_sequence

def load_chroma_full_scratch():
    class MegatronEmbeddings(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            tokenizer = AutoTokenizer.from_pretrained(
                "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                trust_remote_code=True)
            config = AutoConfig.from_pretrained("/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                                                "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                                                trust_remote_code=True)
            model = AutoModel.from_pretrained("/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                                              "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                                              trust_remote_code=True)
            model.to("cuda")
            token_ids = tokenizer(input, return_tensors='pt')['input_ids']
            token_ids = token_ids.to("cuda")
            with torch.no_grad():
                outputs = model(token_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Taking mean over all tokens
                embeddings = embeddings.cpu().squeeze().numpy()
                embeddings = embeddings.tolist()
            return embeddings

    chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="/gscratch3/users/nlebena001/Datasets/MIMIC-IV/Vector_database")
    collection = chroma_client.get_or_create_collection(name="Patients_embeddings_200_labels",
                                                        embedding_function=MegatronEmbeddings())

    return chroma_client, collection

def load_chroma_full_pretrained():
    class MegatronEmbeddings(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            tokenizer = AutoTokenizer.from_pretrained(
                "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                     "Masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                     trust_remote_code=True)
            config = AutoConfig.from_pretrained("/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                     "Masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                     trust_remote_code=True)
            model = AutoModel.from_pretrained("/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                     "Masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000",
                     trust_remote_code=True)
            model.to("cuda")
            token_ids = tokenizer(input, return_tensors='pt')['input_ids']
            token_ids = token_ids.to("cuda")
            with torch.no_grad():
                outputs = model(token_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Taking mean over all tokens
                embeddings = embeddings.cpu().squeeze().numpy()
                embeddings = embeddings.tolist()
            return embeddings

    chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="/gscratch3/users/nlebena001/Datasets/MIMIC-IV/Vector_database")
    collection = chroma_client.get_collection(name="Patients_embeddings_osa_mimic_further_pre_trained",
                                                        embedding_function=MegatronEmbeddings())
    return chroma_client, collection


def load_model_features(model_name):
    if model_name=="ICD-10 full codes scratch":

        model_path = "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                     "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000"

        inference_model_path = "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                               "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained_next_code_pred/" \
                               "model/checkpoint-8000"
        model, tokenizer, inference_model_LM,inference_model_sequence=load_model(model_path,inference_model_path)
        chroma_client,collection=load_chroma_full_scratch()
        return model, tokenizer, inference_model_LM, inference_model_sequence, chroma_client, collection

    else:
        model_path = "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                     "Masked_LM_training_mimic_iv_osa_full_pretrained/model/checkpoint-129000"

        inference_model_path = "/gscratch3/users/nlebena001/Timeline_experiments/Experiments_data/" \
                               "Gatortron_scratch_masked_LM_training_mimic_iv_osa_full_pretrained_next_code_pred/" \
                               "model/checkpoint-8000"
        model, tokenizer, inference_model_LM, inference_model_sequence = load_model(
            model_path, inference_model_path)
        chroma_client,collection=load_chroma_full_pretrained()


        return model, tokenizer, inference_model_LM, inference_model_sequence, chroma_client, collection

def get_explainable_prediction_mixtral(ehr: str, dt: str, icd: str, max_new_tokens: int = 4000, temperature: float = 0.2) -> str:
    """
    Genera una explicación para una predicción usando Mixtral con el modelo
    mistralai/Mixtral-8x7B-Instruct-v0.1. Usa el prompt en español.
    
    Args:
        ehr: Texto del EHR del paciente
        dt: Código de diagnóstico predicho
        icd: Definición del código de diagnóstico
        max_new_tokens: Número máximo de tokens a generar
        temperature: Temperatura para la generación
    
    Returns:
        Texto de la explicación generada
    """
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    prompt_template = (
        "[INST] \n"
        "Eres un experto clínico en historias clínicas electrónicas (EHR).\n\n"
        "Tu objetivo es proporcionar explicaciones plausibles para Predicciones de Próximos Diagnósticos, dado un EHR.\n\n"
        "Recibirás un EHR de un paciente. También recibirás una predicción de un posible Próximo Diagnóstico que el paciente podría tener.\n\n"
        "# Conocimientos\n\n"
        "Este es el EHR del paciente:```{ehr}```\n\n"
        "La predicción experta para el Próximo Diagnóstico del paciente, dado el EHR anterior, es:\n\n"
        "- Próximo Diagnóstico: {dt} - {icd}\n\n"
        "# Directrices\n\n"
        "- Usa referencias del texto\n\n"
        "IMPORTANTE: Explica por qué el próximo diagnóstico es plausible para el paciente, dada su historia clínica.\n\n"
        "IMPORTANTE: No escribas cuál es la tarea, solo da la explicación. No saludes. Empieza con algo como \"El próximo diagnóstico es plausible para el paciente porque...\"\n\n"
        "IMPORTANTE: Escribe en castellano\n\n"
        "IMPORTANTE: NO ESCRIBAS EN INGLÉS\n"
        "# Tarea\n\n"
        "Da una explicación en castellano de por qué el Próximo Diagnóstico es plausible para el paciente, dado el EHR.\n"
        "[/INST]"
    )
    
    model, tokenizer = _get_or_load_mixtral_model(model_id)
    
    # Formatear el prompt con los valores
    formatted_prompt = prompt_template.format(ehr=ehr, dt=dt, icd=icd)
    
    # Si el tokenizer tiene plantilla de chat, usarla; si no, usar prompt directo
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": formatted_prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = formatted_prompt
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # Guardar longitud del input
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=0.9
        )
    # Extraer solo los tokens generados (excluyendo el prompt de entrada)
    generated_tokens = outputs[0][input_length:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text


def markdown_to_html(text):
    """
    Convierte un texto que mezcla HTML y Markdown (con * y + para listas,
    y múltiples espacios como saltos de línea) a un HTML más estándar.
    1) Reemplaza espacios dobles (o más) por <br>.
    2) Convierte líneas que comienzan con '* ' en listas <ul><li>.
    3) Convierte líneas que comienzan con '+ ' en sub-listas anidadas <ul><li>.
    """
    text=text.replace("white-space: pre-wrap","white-space: normal")
    # 1) Reemplazar espacios dobles (o más) por <br>
    text = re.sub(r' {2,}', '<br>', text)

    # 2) Dividir en líneas reales
    lines = text.split('<br>')

    html_output = []
    in_ul = False  # Indica si estamos dentro de una lista principal (<ul>)
    in_sub_ul = False  # Indica si estamos dentro de una sub-lista (<ul> anidada)

    for line in lines:
        stripped = line.strip()

        # -- Manejo de items con '* '
        if stripped.startswith('* '):
            # Si no estamos en una <ul> principal, la abrimos
            if not in_ul:
                html_output.append('<ul>')
                in_ul = True
            # Si había una sub-lista abierta, la cerramos antes de continuar
            if in_sub_ul:
                html_output.append('</ul>')
                in_sub_ul = False

            # El contenido del <li> es lo que sigue al '* '
            content = stripped[2:].strip()
            html_output.append(f'  <li>{content}</li>')

        # -- Manejo de items con '+ ' (sub-lista)
        elif stripped.startswith('+ '):
            # Asegurarnos de que exista una <ul> principal
            if not in_ul:
                html_output.append('<ul>')
                in_ul = True
            # Abrir la sub-lista si no estaba abierta
            if not in_sub_ul:
                html_output.append('  <ul>')
                in_sub_ul = True

            content = stripped[2:].strip()
            html_output.append(f'    <li>{content}</li>')

        else:
            # Si la línea no es un bullet:
            #   Cerrar sub-lista si estaba abierta
            if in_sub_ul:
                html_output.append('  </ul>')
                in_sub_ul = False
            #   Cerrar lista principal si estaba abierta
            if in_ul:
                html_output.append('</ul>')
                in_ul = False

            # Agregar la línea tal cual (manteniendo cualquier HTML existente)
            html_output.append(line)

    # Al final, cerrar listas abiertas si quedaron sin cerrar
    if in_sub_ul:
        html_output.append('  </ul>')
    if in_ul:
        html_output.append('</ul>')

    # Unir todo
    return "\n".join(html_output)