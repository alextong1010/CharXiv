import argparse, json
from openai import OpenAI
from tqdm import tqdm
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--gen_prefix', type=str, default='gen-')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--eval_model', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4o-mini', 'gemma-3-27b-it'], 
                        help='Model to use for evaluation scoring')
    parser.add_argument('--resume_from', type=int, default=None, 
                        help='Resume from a specific query index')
    parser.add_argument('--parse_mode', type=str, default='default', choices=['default', 'parse', 'qcond_parse', 'program_synthesis'],
                        help='Parse mode: "default" for direct question, "parse" to extract JSON first, "qcond_parse" to extract JSON first and then parse with qcond, "program_synthesis" to extract JSON first and then parse with program synthesis')
    args = parser.parse_args()
    
    # Setup client based on eval model
    if args.eval_model == 'gpt-4o':
        client = OpenAI(api_key=args.api_key)
    elif args.eval_model == 'gpt-4o-mini':
        client = OpenAI(api_key=args.api_key)
    elif args.eval_model == 'gemma-3-27b-it':
        from google import genai
        client = genai.Client(api_key=args.api_key)
    else:
        raise ValueError(f"Unsupported eval model: {args.eval_model}")
    
    args.input_file = f"data/{args.mode}_{args.split}.json"
    if args.parse_mode == 'parse':
        args.resp_file = f"results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}_parse.json"
    elif args.parse_mode == 'qcond_parse':
        args.resp_file = f"results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}_qcond_parse.json"
    elif args.parse_mode == 'program_synthesis':
        args.resp_file = f"results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}_program_synthesis.json"
    else:
        args.resp_file = f"results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}.json"
    args.output_file = args.resp_file.replace(args.gen_prefix, "scores-")
    print(f"Output file: {args.output_file}")

    data, response = json.load(open(args.input_file)), json.load(open(args.resp_file))
    mode = 'descriptive' if 'descriptive' in args.resp_file.split('-')[-1] else 'reasoning'

    if mode == 'descriptive':
        from descriptive_utils import preprocess_descriptive_grading_queries, build_descriptive_grading_queries, \
                postprocess_descriptive_grading_queries, get_descriptive_result_gpt, get_descriptive_result_gpt_4o_mini, get_descriptive_result_gemma
        # group the responses based on the template id instead of figure id
        groups = preprocess_descriptive_grading_queries(data, response)
        # batched evaluation based on number of questions per query (nq_per_query)
        queries = build_descriptive_grading_queries(groups)
        combined_queries = []
        
        # Initialize results dictionary for incremental saving
        all_results = {}

        if args.resume_from is not None:
            # Read from previous state
            try:
                all_results = json.load(open(args.output_file))
                # Truncate results to only include items up to resume point
                # Each query processes 5 items, so keep first resume_from * 5 items
                max_items = args.resume_from * 5
                if len(all_results) > max_items:
                    # Get the first max_items keys and create a new dict
                    truncated_keys = list(all_results.keys())[:max_items]
                    all_results = {key: all_results[key] for key in truncated_keys}
                    print(f"Truncated results to {len(all_results)} items (resume_from={args.resume_from})")

            except (FileNotFoundError, json.JSONDecodeError):
                all_results = {}
        
        for i, query in enumerate(tqdm(queries)):
            if args.resume_from is not None and i < args.resume_from:
                continue
            if args.eval_model == 'gpt-4o':
                result = get_descriptive_result_gpt(client, query['grading_query'], len(query['resp_keys']))
            elif args.eval_model == 'gpt-4o-mini':
                result = get_descriptive_result_gpt_4o_mini(client, query['grading_query'], len(query['resp_keys']), i)
            elif args.eval_model == 'gemma-3-27b-it':
                result = get_descriptive_result_gemma(client, query['grading_query'], len(query['resp_keys']))
            # query contains resp_keys, grading_query, extract_answer and score
            combined_queries.append({**query, **result})
            
            # Process and save results incrementally
            temp_queries = postprocess_descriptive_grading_queries(combined_queries)
            all_results.update(temp_queries)
            
            # Save after each query
            with open(args.output_file, "w") as f:
                json.dump(all_results, f, indent=4)
        
        queries = all_results
    
    # NOT IMPLEMENTED YET
    elif mode == 'reasoning':
        raise NotImplementedError("Reasoning mode not implemented yet")
        from reasoning_utils import build_reasoning_grading_queries, get_reasoning_result_gpt, get_reasoning_result_gemma
        # dict of figure_id -> {figure_id, grading_query}
        queries = build_reasoning_grading_queries(data, response) 
        for figure_id, query in tqdm(queries.items()):
            if args.eval_model == 'gpt-4o':
                ext, scr = get_reasoning_result_gpt(client, query['grading_query'])
            elif args.eval_model == 'gpt-4o-mini':
                ext, scr = get_reasoning_result_gpt(client, query['grading_query'])  # Use same function as gpt-4o for now
            elif args.eval_model == 'gemma-3-27b-it':
                ext, scr = get_reasoning_result_gemma(client, query['grading_query'])
            queries[figure_id]['extracted_answer'] = ext
            queries[figure_id]['score'] = scr
            queries[figure_id].pop('grading_query')
            
            # Save after each query
            with open(args.output_file, "w") as f:
                json.dump(queries, f, indent=4)
    else: raise ValueError("Mode not supported")

    # Final save (redundant for descriptive mode, but ensures file is saved for reasoning mode)
    print(f"Final results saved to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(queries, f, indent=4)
