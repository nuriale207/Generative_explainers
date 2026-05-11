import transformers

from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain import LLMChain

def get_prediction_explanations(cat2def_dict, result_labels,ehr_text,model, tokenizer,language="english",model_name=""):
    explanations=[]
    text=""
    for label in result_labels:
        text=text+"### Explicación para la predicción:" + label + " " + cat2def_dict[label.upper()]
        text=text+"\n"
        expl_pred=get_explainable_prediction(model, tokenizer, ehr_text,label, cat2def_dict[label.upper()],language=language,model_name=model_name)
        text=text+expl_pred["text"]+"\n\n"
        explanations.append(expl_pred["text"])
    return text,explanations


def get_explainable_prediction(model, tokenizer, ehr, dt, icd,language="english",model_name=""):

    if language=="spanish":
        prompt_template = """
        [INST] 
        Eres un experto clínico en historias clínicas electrónicas (EHR)...
        
        Tu objetivo es proporcionar explicaciones plausibles para Predicciones de Próximos Diagnósticos, dado un EHR.
        
        Recibirás un EHR de un paciente. También recibirás una predicción de un posible Próximo Diagnóstico que el paciente podría tener.
        
        # Conocimientos
        
        Este es el EHR del paciente:```{ehr}```
        
        La predicción experta para el Próximo Diagnóstico del paciente, dado el EHR anterior, es:
        
        - Próximo Diagnóstico: {dt} - {icd}
        
        # Directrices
        
        - Usa referencias del texto
        
        IMPORTANTE: Explica por qué el próximo diagnóstico es plausible para el paciente, dada su historia clínica.
        
        IMPORTANTE: No escribas cuál es la tarea, solo da la explicación. No saludes. Empieza con algo como "El próximo diagnóstico es plausible para el paciente porque..."
        
        IMPORTANTE: Escribe en castellano
        
        IMPORTANTE: NO ESCRIBAS EN INGLÉS 
        # Tarea
        
        Da una explicación en castellano de por qué el Próximo Diagnóstico es plausible para el paciente, dado el EHR.
    
    [/INST]
        """
        if model_name=="BioMistral/BioMistral-7B":
            prompt_template= """
    [INST] 
        Eres un experto clínico en registros de salud electrónicos (EHR).

        Tu objetivo es proporcionar explicaciones plausibles para las Predicciones Diagnósticas Próximas, dado un EHR.
        
        Se te proporcionará un EHR de un paciente, así como una predicción de un posible Próximo Diagnóstico que podría tener el paciente.
        
        Conocimiento
        Este es el EHR del paciente: {ehr}
        
        La predicción experta para el Próximo Diagnóstico del paciente, dado el EHR anterior, es:
        
        Próximo Diagnóstico: {dt} - {icd}
        Directrices
        Usa referencias del texto.
        IMPORTANTE: Explica por qué el próximo diagnóstico es plausible para el paciente, considerando su historial.
        IMPORTANTE: Escribe en castellano.
        
        Tarea
        Proporciona una explicación de por qué el Próximo Diagnóstico es plausible para el paciente, dado el EHR.
    [/INST]
     """

    else:
        prompt_template = """
        [INST] 
        You are a clinical expert in EHRs...
    
        Your objective is to give plausible explanations for Next Diagnostic Predictions, given an EHR.
    
        You will get an EHR from a patient. You will also get a prediction of a possible Next Diagnostic that the patient will have.
    
        # Knowledge
    
        This is the EHR from the patient:```{ehr}```
    
        The expert prediction for the Next Diagnostic for the patient given the above EHR is:
    
        - Next Diagnostic: {dt} - {icd}
    
        # Guidelines
    
        - Use references from the text
    
        IMPORTANT: Explain why the next diagnostic is plausible for the patient given his history
        # Task
    
        Give an explanation of why the Next Diagnostic is plausible for the patient, given the EHR.
        [/INST]
         """
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1,
        return_full_text=False,
        max_new_tokens=4000,
    )

    if language=="spanish" and model_name == "BioMistral/BioMistral-7B":
        text_generation_pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1,
            return_full_text=False,
            max_new_tokens=4000,
            device_map="auto"
        )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["ehr", "dt", "icd"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
    result = llm_chain.invoke({"ehr": ehr,
                               "dt": dt,
                               "icd": icd}, return_only_outputs=True)

    return result