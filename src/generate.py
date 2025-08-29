import os
import json
import argparse
from generate_lib.utils import get_generate_fn, get_client_fn, generate_response_remote_wrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--data_dir', type=str, required=False, default="./data", \
                        help="Directory containing the input json files")
    parser.add_argument('--image_dir', type=str, required=False, default="/n/netscratch/ydu_lab/Lab/alex/CharXiv/images", \
                        help="Directory containing the images")
    parser.add_argument('--output_dir', type=str, required=False, default="./results", \
                        help="Directory to save the output json files")
    parser.add_argument('--split', type=str, required=False, choices=['val', 'test'], default='val',
                        help="Split of the data")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['descriptive', 'reasoning'],
                        help="Mode of the evaluation")
    parser.add_argument('--parse_mode', type=str, required=False, choices=['default', 'parse', 'qcond_parse', 'program_synthesis'], default='default',
                        help="Parse mode: 'default' for direct question, 'parse' to extract JSON first, 'synthesis' for program synthesis")

    # custom arguments
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_api', type=str, nargs='*', required=False, default=None,
                        help="API key(s) for the model. Provide multiple keys for parallel execution.")
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    if args.parse_mode == 'parse':
        output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}_parse.json')
    elif args.parse_mode == 'qcond_parse':
        output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}_qcond_parse.json')
    elif args.parse_mode == 'program_synthesis':
        output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}_program_synthesis.json')
    else:
        output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}.json')

    if args.mode == 'descriptive':
        from descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, args.image_dir)
    elif args.mode == 'reasoning':
        from reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, args.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)

    generate_fn = get_generate_fn(args.model_path)
    
    if args.model_api is not None and len(args.model_api) > 0:
        # Handle both single and multiple API keys
        api_keys = args.model_api
        clients = []
        models = []
        for api_key in api_keys:
            client, model = get_client_fn(args.model_path)(args.model_path, api_key)
            clients.append(client)
            models.append(model)
        # Use the first model for the call (all should be the same)
        generate_response_remote_wrapper(generate_fn, queries, models[0], api_keys, clients, parse_mode=args.parse_mode)
    else:
        generate_fn(queries, args.model_path)

    # for k in queries:
    #     queries[k].pop("figure_path", None)
    #     queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")
