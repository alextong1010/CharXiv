import base64
import requests

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using GPT"
    assert model_path is not None, "Model name is required for using GPT"
    model = model_path
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False, parse_mode='default'):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def make_api_call(prompt_text, include_image=True):
        content = [{"type": "text", "text": prompt_text}]
        if include_image and not random_baseline:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        return response['choices'][0]['message']['content']

    # Common setup
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if parse_mode == 'parse':
        parse_prompt = """Please analyze this image and convert it into a detailed JSON representation that captures all the visual information, including:
        - Text content (equations, numbers, labels, etc.)
        - Geometric shapes and their properties
        - Spatial relationships between elements
        - Colors, sizes, and other visual attributes
        - Any diagrams, charts, graphs, or mathematical figures
        - All relevant visual details that would be needed to answer questions about this image.

        Provide the response as a well-structured JSON object."""
        
        parse_text = make_api_call(parse_prompt, include_image=True)
        final_prompt = f"Based on this JSON representation of an image:\n\n{parse_text}\n\nPlease answer the following question: {query}"
        response = make_api_call(final_prompt, include_image=False)
        
        return {
            'response': response,
            'json_parse': parse_text
        }
    
    elif parse_mode == 'qcond_parse':
        parse_prompt = f"""You are given an image and the following question: \n{query}\n
    Please analyze this image and convert it into a detailed JSON representation that captures all the visual information needed to answer this question, including:
        - Text content (equations, numbers, labels, etc.)
        - Geometric shapes and their properties
        - Spatial relationships between elements
        - Colors, sizes, and other visual attributes
        - Any diagrams, charts, graphs, or mathematical figures
        - All relevant visual details that would be needed to answer questions about this image.

    Provide the response as a well-structured JSON object."""
        
        parse_text = make_api_call(parse_prompt, include_image=True)
        final_prompt = f"Based on this JSON representation of an image:\n\n{parse_text}\n\nPlease answer the following question: {query}"
        response = make_api_call(final_prompt, include_image=False)
        
        return {
            'response': response,
            'json_parse': parse_text
        }
    
    elif parse_mode == 'program_synthesis':
        synthesis_prompt = """Please analyze this chart/figure and generate a program in a charting domain-specific language (DSL) that could be used to programmatically recreate this exact visualization. Focus on creating a complete, executable specification that includes:
        - Chart type (bar, line, pie, scatter, etc.)
        - Data values and series
        - Axis labels, titles, and scales
        - Colors, styling, and visual properties
        - Layout and positioning information
        - Any annotations, legends, or additional elements

        Provide the response as a well-structured program written in the DSL, with explicit function calls and parameters that a charting engine could execute to recreate this visualization."""
        
        synthesis_text = make_api_call(synthesis_prompt, include_image=True)
        final_prompt = f"Based on this program specification of an image:\n\n{synthesis_text}\n\nPlease answer the following question: {query}"
        response = make_api_call(final_prompt, include_image=False)
        
        return {
            'response': response,
            'json_synthesis': synthesis_text
        }
    
    else:
        # Default mode: existing behavior
        return make_api_call(query, include_image=True)