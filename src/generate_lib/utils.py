import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

def process_queries_subset(generate_fn, queries_subset, model_path, api_key, client, 
                          init_sleep=1, max_retries=20, sleep_factor=1.6, parse_mode='default'):
    """Process a subset of queries sequentially"""
    for k in tqdm(queries_subset, desc=f"Processing queries"):
        sleep_time = init_sleep
        query = queries_subset[k]['question']
        image = queries_subset[k]["figure_path"]
        curr_retries = 0
        result = None
        while curr_retries < max_retries and result is None:
            try:
                result = generate_fn(image, query, model_path, 
                    api_key=api_key, client=client, random_baseline=False, parse_mode=parse_mode)
            except Exception as e:
                print(f"Error for query {k}: {e}")
                print(f"Error {curr_retries}, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                curr_retries += 1
                sleep_time *= sleep_factor
        if result is None:
            result = "Error in generating response."
            print(f"Error in generating response for {k}")
        
        # Handle different return types based on parse_mode
        if parse_mode == 'parse' and isinstance(result, dict):
            queries_subset[k]['response'] = result['response']
            queries_subset[k]['json_parse'] = result['json_parse']
        elif parse_mode == 'qcond_parse' and isinstance(result, dict):
            queries_subset[k]['response'] = result['response']
            queries_subset[k]['json_parse'] = result['json_parse']
        elif parse_mode == 'program_synthesis' and isinstance(result, dict):
            queries_subset[k]['response'] = result['response']
            queries_subset[k]['json_synthesis'] = result['json_synthesis']
        else:
            queries_subset[k]['response'] = result

def generate_response_remote_wrapper(generate_fn, 
        queries, model_path, api_keys, clients, init_sleep=1, 
        max_retries=20, sleep_factor=1.6, parse_mode='default'):
    # Handle both single and multiple API keys for backward compatibility
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    if not isinstance(clients, list):
        clients = [clients]
    
    # If only one API key, process sequentially
    if len(api_keys) == 1:
        process_queries_subset(generate_fn, queries, model_path, api_keys[0], clients[0],
                             init_sleep, max_retries, sleep_factor, parse_mode)
        return
    
    # Split queries into two halves for parallel processing
    query_items = list(queries.items())
    mid_point = len(query_items) // 2
    
    # Create two subsets
    first_half = dict(query_items[:mid_point])
    second_half = dict(query_items[mid_point:])
    
    # Use first two API keys and clients
    api_key1, api_key2 = api_keys[0], api_keys[1]
    client1, client2 = clients[0], clients[1]
    
    # Process both halves in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(process_queries_subset, generate_fn, first_half, model_path, 
                                api_key1, client1, init_sleep, max_retries, sleep_factor, parse_mode)
        future2 = executor.submit(process_queries_subset, generate_fn, second_half, model_path, 
                                api_key2, client2, init_sleep, max_retries, sleep_factor, parse_mode)
        
        # Wait for both to complete
        future1.result()
        future2.result()
    
    # Results are already updated in-place in the query subsets
    # Preserve original order by updating queries with results in original key order
    original_keys = list(queries.keys())
    for key in original_keys:
        if key in first_half:
            queries[key] = first_half[key]
        elif key in second_half:
            queries[key] = second_half[key]

def get_client_fn(model_path):
    if model_path in ['claude-3-sonnet-20240229', 
                      'claude-3-opus-20240229', 
                      'claude-3-haiku-20240307', 
                      'claude-3-5-sonnet-20240620']:
        from .claude import get_client_model
    # gemini
    elif model_path in ['gemini-1.5-pro-001', 
                        'gemini-1.0-pro-vision-001', 
                        'gemini-1.5-flash-001',
                        'gemini-1.5-pro-exp-0801']:
        from .gemini import get_client_model
    elif model_path in ['models/gemma-3-4b-it',
                        'models/gemma-3-12b-it',
                        'models/gemma-3-27b-it']:
        from .gemma3 import get_client_model    
    # gpt
    elif model_path in ['gpt-4o-2024-05-13', 
                        'gpt-4o-2024-08-06',
                        'chatgpt-4o-latest',
                        'gpt-4-turbo-2024-04-09', 
                        'gpt-4o-mini-2024-07-18']:
        from .gpt import get_client_model
    # o1
    elif model_path in ['o1-preview',
                        'o1-mini',
                        'o1-2024-12-17']:
        from .o1 import get_client_model
    # reka
    elif model_path in ['reka-core-20240415', 
                        'reka-flash-20240226', 
                        'reka-core-20240415']:
        from .reka import get_client_model
    # qwen
    elif model_path in ['qwen-vl-max', 
                        'qwen-vl-plus']:
        from .qwen import get_client_model
    # internvl2pro
    elif model_path in ['InternVL2-Pro']:
        from .internvl2pro import get_client_model
    else:
        raise ValueError(f"Model {model_path} not supported")
    return get_client_model

def get_generate_fn(model_path):
    model_name = model_path.split('/')[-1]
    # cambrian
    if model_name in ['cambrian-34b']:
        from .cambrian import generate_response
    # chartgemma
    elif model_name in ['chartgemma']:
        from .chartgemma import generate_response
    # claude
    elif model_name in ['claude-3-sonnet-20240229',
                        'claude-3-opus-20240229',
                        'claude-3-haiku-20240307',
                        'claude-3-5-sonnet-20240620']:
        from .claude import generate_response
    # llama 3.2
    elif model_name in ['Llama-3.2-11B-Vision-Instruct',
                        'Llama-3.2-90B-Vision-Instruct']:
        from .llama32 import generate_response
    # llavaov
    elif model_name in ['llava-onevision-qwen2-0.5b-ov',
                        'llava-onevision-qwen2-7b-ov',
                        'llava-onevision-qwen2-72b-ov-chat']:
        from .llavaov import generate_response
    # molmo
    elif model_name in ['Molmo-7B-D-0924',
                        'Molmo-7B-O-0924',
                        'Molmo-72B-0924',
                        'MolmoE-1B-0924',]:
        from .molmo import generate_response
    # nvlm
    elif model_name in ['NVLM-D-72B']:
        from .nvlm import generate_response
    # phi35
    elif model_name in ['Phi-3.5-vision-instruct']:
        from .phi35 import generate_response
    # pixtral
    elif model_name in ['Pixtral-12B-2409']:
        from .pixtral import generate_response
    # qwen2
    elif model_name in ['Qwen2-VL-2B-Instruct',
                        'Qwen2-VL-7B-Instruct',
                        'Qwen2-VL-72B-Instruct']:
        from .qwen2 import generate_response
    # deepseekvl
    elif model_name in ['deepseek-vl-7b-chat']:
        from .deepseekvl import generate_response
    # gemini
    elif model_name in ['gemini-1.5-pro-001', 
                        'gemini-1.0-pro-vision-001', 
                        'gemini-1.5-flash-001',
                        'gemini-1.5-pro-exp-0801']:
        from .gemini import generate_response
    elif model_name in ['gemma-3-4b-it',
                        'gemma-3-12b-it',
                        'gemma-3-27b-it']:
        from .gemma3 import generate_response
    # gpt
    elif model_name in ['gpt-4o-2024-05-13', 
                        'gpt-4o-2024-08-06',
                        'chatgpt-4o-latest',
                        'gpt-4-turbo-2024-04-09', 
                        'gpt-4o-mini-2024-07-18']:
        from .gpt import generate_response
    # o1
    elif model_name in ['o1-preview',
                        'o1-mini',
                        'o1-2024-12-17']:
        from .o1 import generate_response
    # idefics2
    elif model_name in ['idefics2-8b',
                        'idefics2-8b-chatty',
                        'Idefics3-8B-Llama3']:
        from .idefics import generate_response
    # ixc2
    elif model_name in ['internlm-xcomposer2-4khd-7b',
                        'internlm-xcomposer2-vl-7b']:
        from .ixc2 import generate_response
    # internvl2
    elif model_name in ['InternVL2-26B',
                        'InternVL2-Llama3-76B']:
        from .internvl2 import generate_response
    # internvl15
    elif model_name in ['InternVL-Chat-V1-5']:
        from .internvl15 import generate_response
    # llava16
    elif model_name in ['llava-v1.6-34b-hf',
                        'llava-v1.6-mistral-7b-hf']:
        from .llava16 import generate_response
    # mgm
    elif model_name in ['MGM-34B-HD',
                        'MGM-8B-HD']:
        from .mgm import generate_response
    # minicpm
    elif model_name in ['MiniCPM-Llama3-V-2_5',
                        'MiniCPM-V-2',
                        'MiniCPM-V-2_6']:
        from .minicpm import generate_response
    elif model_name in ['glm-4v-9b']:
        from .glm import generate_response
    # moai
    elif model_name in ['MoAI-7B']:
        from .moai import generate_response
    # paligemma
    elif model_name in ['paligemma-3b-mix-448']:
        from .paligemma import generate_response
    # phi3
    elif model_name in ['Phi-3-vision-128k-instruct']:
        from .phi3 import generate_response
    # qwen
    elif model_name in ['qwen-vl-max',
                        'qwen-vl-plus']:
        from .qwen import generate_response
    # reka
    elif model_name in ['reka-core-20240415',
                        'reka-flash-20240226',
                        'reka-core-20240415']:
        from .reka import generate_response
    # sphinx
    elif model_name in ['SPHINX-v2-1k']:
        from .sphinx2 import generate_response
    # vila
    elif model_name in ['VILA1.5-40b']:
        from .vila15 import generate_response
    # ovis
    elif model_name in ['Ovis1.5-Llama3-8B',
                        'Ovis1.5-Gemma2-9B']:
        from .ovis import generate_response
    # internvl2pro
    elif model_name in ['InternVL2-Pro']:
        from .internvl2pro import generate_response
    elif model_name in ['ChartLlama-13b']:
        from .chartllama import generate_response
    elif model_name in ['TinyChart-3B-768']:
        from .tinychart import generate_response
    elif model_name in ['ChartInstruct-LLama2',
                        'ChartInstruct-FlanT5-XL']:
        from .chartinstruct import generate_response
    elif model_name in ['unichart-chartqa-960']:
        from .unichart import generate_response
    elif model_name in ['ChartAssistant']:
        from .chartast import generate_response
    elif model_name in ['DocOwl1.5-Omni',
                        'DocOwl1.5-Chat',]:
        from .docowl15 import generate_response
    elif model_name in ['ureader-v1']:
        from .ureader import generate_response
    elif model_name in ['TextMonkey',
                        'Monkey-Chat',]:
        from .textmonkey import generate_response
    elif model_name in ['cogagent-vqa-hf']:
        from .cogagent import generate_response
    else:
        raise ValueError(f"Model {model_name} not supported")
    return generate_response
