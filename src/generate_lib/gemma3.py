from PIL import Image
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
import time

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Gemma"
    assert model_path is not None, "Model name is required for using Gemma"
    client = genai.Client(api_key=api_key)
    model = model_path  # Return model_path as model (CharXiv convention)
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False, parse_mode='default'):
    # model parameter is the model_path string (CharXiv convention)
    # client is the genai.Client instance
    # time.sleep(0.5) # to help with rate limiting
    image = Image.open(image_path)
    
    if parse_mode == 'parse':
        # First, extract JSON representation from the image
        parse_prompt = """Please analyze this image and convert it into a detailed JSON representation that captures all the visual information, including:
        - Text content (equations, numbers, labels, etc.)
        - Geometric shapes and their properties
        - Spatial relationships between elements
        - Colors, sizes, and other visual attributes
        - Any diagrams, charts, graphs, or mathematical figures
        - All relevant visual details that would be needed to answer questions about this image.

        Provide the response as a well-structured JSON object."""
        parse_response = client.models.generate_content(
            model=model,
            contents=[image, parse_prompt],
        )
        
        # Then ask the original question using the JSON representation
        final_prompt = f"Based on this JSON representation of an image:\n\n{parse_response.text}\n\nPlease answer the following question: {query}"
        response = client.models.generate_content(
            model=model,
            contents=[final_prompt],
        )
        
        # Return both the final response and the JSON parse output
        return {
            'response': response.text,
            'json_parse': parse_response.text
        }
    
    elif parse_mode == 'qcond_parse':
        # First, extract JSON representation from the image
        parse_prompt = f"""You are given an image and the following question: \n{query}\n
    Please analyze this image and convert it into a detailed JSON representation that captures all the visual information needed to answer this question, including:
        - Text content (equations, numbers, labels, etc.)
        - Geometric shapes and their properties
        - Spatial relationships between elements
        - Colors, sizes, and other visual attributes
        - Any diagrams, charts, graphs, or mathematical figures
        - All relevant visual details that would be needed to answer questions about this image.

    Provide the response as a well-structured JSON object."""
        parse_response = client.models.generate_content(
            model=model,
            contents=[image, parse_prompt],
        )
        
        # Then ask the original question using the JSON representation
        final_prompt = f"Based on this JSON representation of an image:\n\n{parse_response.text}\n\nPlease answer the following question: {query}"
        response = client.models.generate_content(
            model=model,
            contents=[final_prompt],
        )
        
        # Return both the final response and the JSON parse output
        return {
            'response': response.text,
            'json_parse': parse_response.text
        }
    
    elif parse_mode == 'program_synthesis':
        # First, generate JSON specification that would create this chart/figure
        synthesis_prompt = """Please analyze this chart/figure and generate a program in a charting domain-specific language (DSL) that could be used to programmatically recreate this exact visualization. Focus on creating a complete, executable specification that includes:
        - Chart type (bar, line, pie, scatter, etc.)
        - Data values and series
        - Axis labels, titles, and scales
        - Colors, styling, and visual properties
        - Layout and positioning information
        - Any annotations, legends, or additional elements

        Provide the response as a well-structured program written in the DSL, with explicit function calls and parameters that a charting engine could execute to recreate this visualization."""
        synthesis_response = client.models.generate_content(
            model=model,
            contents=[image, synthesis_prompt],
        )
        
        # Then ask the original question using the JSON specification
        final_prompt = f"Based on this program specification of an image:\n\n{synthesis_response.text}\n\nPlease answer the following question: {query}"
        response = client.models.generate_content(
            model=model,
            contents=[final_prompt],
        )
        
        # Return both the final response and the JSON synthesis output
        return {
            'response': response.text,
            'json_synthesis': synthesis_response.text
        }
        
    else:
        # Default mode: ask question directly on the image
        response = client.models.generate_content(
            model=model,  # model is the model_path string
            contents=[image, query],
        )
    return response.text