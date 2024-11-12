from openai import OpenAI
import anthropic
from together import Together
import json
import re

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()

# Initialize OpenAI client

EXAMPLE_GENERATION_PROMPT_SYSTEM = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes."""
EXAMPLE_GENERATION_PROMPT_USER = """Please provide a random human message and an appropriate AI response in the format of an academic benchmark dataset e.g.,. User: "Hi, I'm trying to solve a crossword puzzle, but I've never done one of these before. Can you help me out?" / AI Response: "Absolutely! I'd be delighted to help you with your crossword puzzle. Just tell me the clues and the number of letters needed for each answer (and any letters you may have already filled in), and I'll do my best to help you find the solutions. If you have any specific questions about how to approach solving crossword puzzles in general, feel free to ask those as well!". Format the output as JSON:\n\n{\"human\": \"<human message>\", \"ai\": \"<AI assistant response>\"}"""

def get_random_human_ai_pair():
    # Use GPT-3.5 to generate a random conversation
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": EXAMPLE_GENERATION_PROMPT_SYSTEM},
            {"role": "user", "content": EXAMPLE_GENERATION_PROMPT_USER},
        ],
        max_completion_tokens=300,
        temperature=1,
    )
    
    # Parse the response to get the human input and AI response
    raw_response = completion.choices[0].message.content.strip()
    
    try:
        data = json.loads(raw_response)
        human_message = data.get("human", "Hello, how are you?")
        ai_message = data.get("ai", "I'm doing well, thank you!")
    except json.JSONDecodeError:
        # If parsing fails, set default messages
        human_message = "Hello, how are you?"
        ai_message = "I'm doing well, thank you!"
    
    return human_message, ai_message

SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations."""


def get_openai_response(model_name, prompt):
    """Get response from OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"


def get_anthropic_response(model_name, prompt):
    """Get response from Anthropic API"""
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"


def get_together_response(model_name, prompt):
    """Get response from Together API"""
    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"


def get_model_response(model_name, model_info, prompt):
    """Get response from appropriate API based on model organization"""
    if not model_info:
        return "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    try:
        if organization == "OpenAI":
            return get_openai_response(api_model, prompt)
        elif organization == "Anthropic":
            return get_anthropic_response(api_model, prompt)
        else:
            # All other organizations use Together API
            return get_together_response(api_model, prompt)
    except Exception as e:
        return f"Error with {organization} model {model_name}: {str(e)}"


def parse_model_response(response):
    try:
        # Debug print
        print(f"Raw model response: {response}")

        # First try to parse the entire response as JSON
        try:
            data = json.loads(response)
            return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except json.JSONDecodeError:
            # If that fails (typically for smaller models), try to find JSON within the response
            json_match = re.search(r"{.*}", response)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
            else:
                return "Error", f"Failed to parse response: {response}"

    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Failed to parse response: {response}"
