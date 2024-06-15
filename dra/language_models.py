# Standard Library Imports
import os

# Third-Party Imports
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local Imports
from dra.utils import log_yellow, LLAMA2_PROMPT_LONG, LLAMA2_PROMPT_SHORT, VICUNA_PROMPT


openai.api_key = os.getenv("OPENAI_API_KEY")

def load_model(target_model):
    log_yellow('[*] Loading target model...')
    model_path = get_model_path(target_model)
    model_kwargs = {"low_cpu_mem_usage": True, "use_cache": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    log_yellow('[*] Target model loaded!')
    if 'llama' in target_model.lower():
        conv_prompt = LLAMA2_PROMPT_LONG['prompt']
    elif 'vicuna' in target_model.lower():
        conv_prompt = VICUNA_PROMPT['prompt']
    else:
        # not supported model
        raise NotImplementedError
    return tokenizer, model, conv_prompt



def chat_with_gpt(prompt, model=None):
    if '3.5' in model:
        model = 'gpt-3.5-turbo-0613'
    elif 'gpt-4' in model:
        model = 'gpt-4-0613'

    temperature=0.0
    n=1
    max_trial = 50
    for _ in range(max_trial):
        try:
            response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        n=n,
                        max_tokens=256,
                    )
            break
        except Exception as e:
            print(e)
            time.sleep(5)
            continue

    return response.choices[0].message['content']


def chat_with_opensource(prompt, tokenizer, model, conv_prompt):
    test_cases_formatted = conv_prompt.format(instruction=prompt)
    inputs = tokenizer([test_cases_formatted], return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda')
    num_input_tokens = input_ids.shape[1]
    outputs = model.generate(input_ids, max_new_tokens=400, do_sample=False)
    response = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)
    return response


def get_model_path(model_name):
    full_model_dict={
        "llama2-7b": {
            "model_path": "/media/d1/huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
            },
        "llama2-13b": {
            "model_path": "/media/d1/huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"
            },
        "llama2-70b": {
            "model_path": "/media/d1/huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
            },
        "zephyr": {
            "model_path": "/media/d1/huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
            },
        "mixtral": {
            "model_path": "/media/d1/huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
            },
        "nh2sft": {
            "model_path": "/media/d1/huggingface.co/models/Nous-Hermes-2-Mixtral-8x7B-SFT"
            },
        "nh2dpo": {
            "model_path": "/media/d1/huggingface.co/models/Nous-Hermes-2-Mixtral-8x7B-DPO"
            },
        "mistral": {
            "model_path": "/media/d1/huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            },
        "vicuna": {
            "model_path": "/media/d1/huggingface.co/models/lmsys/vicuna-13b-v1.5"
            },
        "harmbench": {
            "model_path": "/media/d1/huggingface.co/models/cais/HarmBench-Llama-2-13b-cls"
        },
        "/media/d1/huggingface.co/models/meta-llama/Llama-2-13b-chat-hf": {
            "model_path": "/media/d1/huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"
        },
        "/media/d1/huggingface.co/models/cais/HarmBench-Llama-2-13b-cls": {
            "model_path": "/media/d1/huggingface.co/models/cais/HarmBench-Llama-2-13b-cls"
        }
    }
    path = full_model_dict[model_name]["model_path"]
    return path
