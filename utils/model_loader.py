import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM


def load_quantified_model(model_id):
    # model_id = "BioMistral/BioMistral-7B"
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


def get_prediction_labels(prediction_list, id2label):
    # print(id2label)
    # print(prediction_list)
    #
    # print(len(prediction_list))
    etiquetas_activas = []
    # Iterar sobre cada predicción y su índice
    # print(len(prediction_list))
    for idx, pred in enumerate(prediction_list):
        if pred == 1:
            etiquetas_activas.append(id2label[idx])
    return etiquetas_activas
