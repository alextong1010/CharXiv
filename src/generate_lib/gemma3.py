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

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    # model parameter is the model_path string (CharXiv convention)
    # client is the genai.Client instance
    time.sleep(1) # to help with rate limiting
    image = Image.open(image_path)
    response = client.models.generate_content(
        model=model,  # model is the model_path string
        contents=[image, query],
    )
    return response.text