from openai import OpenAI
import anthropic
from together import Together
import json
import re
import random

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()

GOOD_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The AI response generated should be longer than a few sentences long."""
BAD_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The response should contain incorrect information, logical fallacies, or misleading explanations. It should sound plausible but be fundamentally wrong. The AI response generated should be longer than a few sentences long."""
AMBIGUOUS_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The response should mix correct and incorrect information - it should contain some accurate points but also include nuanced, questionable claims or exaggerations. The AI response generated should be longer than a few sentences long."""

GENERATION_PROMPT = """Please generate a human message and an AI response in the format of a QA dataset. The AI response generated should be at least a few sentences long.

Format your output as JSON:\n\n{\"human\": \"<human message>\", \"ai\": \"<AI assistant response>\"}"""

RESPONSE_GENERATION_SYSTEM_PROMPT = "You are an assistant that generates random responses to human messages for testing purposes. Generate bad responses (with a mix of correct and incorrect information) 60% of the time and good responses 40% of the time. Do not say which type of response you are generating, just generate the response."
def get_random_human_ai_pair():
    # Select system prompt with specified probabilities
    system_prompt = random.choices(
        [GOOD_SYSTEM_PROMPT, BAD_SYSTEM_PROMPT, AMBIGUOUS_SYSTEM_PROMPT],
        weights=[0.2, 0.2, 0.6]  # 20% good, 20% bad, 60% ambiguous
    )[0]
    
    # Log which type of response is being generated
    prompt_type = {
        GOOD_SYSTEM_PROMPT: "good",
        BAD_SYSTEM_PROMPT: "bad", 
        AMBIGUOUS_SYSTEM_PROMPT: "ambiguous"
    }[system_prompt]
    print(f"Generating {prompt_type} response")
    
    # Randomly choose between GPT-3.5 and Claude
    model_choice = random.choice([
        ("gpt-3.5-turbo", get_openai_response),
        ("claude-3-5-haiku-latest", get_anthropic_response)
    ])
    model_name, api_func = model_choice
    
    # Generate response using selected model
    response = api_func(
        model_name=model_name,
        prompt=GENERATION_PROMPT,
        system_prompt=system_prompt,
        max_tokens=600,
        temperature=1
    )
    
    # Parse the response to get the human input and AI response
    try:
        data = json.loads(response) 
        human_message = data.get("human", """How do muscles grow?""")
        ai_message = data.get("ai", """Muscles grow through a process called skeletal muscle hypertrophy, which adds more myosin filaments to each muscle fiber, making the engine of the cell bigger and stronger over time. This is achieved through increased muscle tension and physical stress, breaking down muscle fiber[3]. Muscle growth is also a direct consequence of resistance training and nutrition. People build muscle at different rates depending on their age, sex, and genetics, but muscle development significantly increases if exercise is done correctly and the body stores more protein through a process called protein synthesis.""")
    except json.JSONDecodeError:
        # If parsing fails, set default messages
        human_message = "Hello, how are you?"
        ai_message = "I'm doing well, thank you!"
    
    return human_message, ai_message

JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations."""


def get_openai_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI model {model_name}: {str(e)}"


def get_anthropic_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from Anthropic API"""
    try:
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error with Anthropic model {model_name}: {str(e)}"


def get_together_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from Together API"""
    try:
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Together model {model_name}: {str(e)}"


def get_model_response(model_name, model_info, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from appropriate API based on model organization"""
    if not model_info:
        return "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    try:
        if organization == "OpenAI":
            return get_openai_response(api_model, prompt, system_prompt, max_tokens, temperature)
        elif organization == "Anthropic":
            return get_anthropic_response(api_model, prompt, system_prompt, max_tokens, temperature)
        else:
            # All other organizations use Together API
            return get_together_response(api_model, prompt, system_prompt, max_tokens, temperature)
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
            json_match = re.search(r"{.*}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
            else:
                return "Error", f"Invalid response format returned - here is the raw model response: {response}"

    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Failed to parse response: {response}"
    
def generate_ai_response(human_msg):
    """Generate AI response using GPT-3.5-turbo"""
    if not human_msg.strip():
        return "", False
        
    try:
        response = get_openai_response(
            "gpt-3.5-turbo", 
            human_msg,
            system_prompt=RESPONSE_GENERATION_SYSTEM_PROMPT,
            max_tokens=1000,
            temperature=1
        )
        # Extract just the response content since we don't need JSON format here
        if isinstance(response, str):
            # Clean up any JSON formatting if present
            try:
                data = json.loads(response)
                response = data.get("content", response)
            except json.JSONDecodeError:
                pass
        return response, False  # Return response and button interactive state
    except Exception as e:
        return f"Error generating response: {str(e)}", False