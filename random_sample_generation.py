from openai import OpenAI
import anthropic
import json
import re
import random
import os
from gen_api_answer import get_openai_response, get_anthropic_response

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()

GOOD_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The AI response generated should be a few sentences long. Format your output as JSON: {"human": "<human message>", "ai": <AI assistant response>}. Ensure the output is valid JSON, without additional formatting or explanations."""
BAD_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The response should contain incorrect information, logical fallacies, or misleading explanations. It should sound plausible but be fundamentally wrong. The AI response generated should be a few sentences long. Format your output as JSON: {"human": "<human message>", "ai": <AI assistant response>}. Ensure the output is valid JSON, without additional formatting or explanations."""
AMBIGUOUS_SYSTEM_PROMPT = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes. The response should mix correct and incorrect information - it should contain some accurate points but also include nuanced, questionable claims or exaggerations. The AI response generated should be a few sentences long. Format your output as JSON: {"human": "<human message>", "ai": <AI assistant response>}. Ensure the output is valid JSON, without additional formatting or explanations."""

GOOD_SYSTEM_PROMPT_WITH_GROUND_TRUTH = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes, along with an ideal reference answer. The AI response generated should be a few sentences long and contain accurate information. The ground truth response should be a perfect, comprehensive answer that would score 5/5. Format your output as JSON: {"human": "<human message>", "ai": "<AI assistant response>", "ground_truth": "<perfect reference answer>"}. Ensure the output is valid JSON, without additional formatting or explanations."""
BAD_SYSTEM_PROMPT_WITH_GROUND_TRUTH = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes, along with an ideal reference answer. The AI response should be a few sentences long and contain incorrect information, logical fallacies, or misleading explanations. It should sound plausible but be fundamentally wrong. The ground truth response should be a perfect, comprehensive answer that would score 5/5. Format your output as JSON: {"human": "<human message>", "ai": "<AI assistant response>", "ground_truth": "<perfect reference answer>"}. Ensure the output is valid JSON, without additional formatting or explanations."""
AMBIGUOUS_SYSTEM_PROMPT_WITH_GROUND_TRUTH = """You are an assistant that generates random conversations between a human and an AI assistant for testing purposes, along with an ideal reference answer. The AI response should be a few sentences long and mix correct and incorrect information - it should contain some accurate points but also include nuanced, questionable claims or exaggerations. The ground truth response should be a perfect, comprehensive answer that would score 5/5. Format your output as JSON: {"human": "<human message>", "ai": "<AI assistant response>", "ground_truth": "<perfect reference answer>"}. Ensure the output is valid JSON, without additional formatting or explanations."""

GENERATION_PROMPT = """Please generate a random human message and an AI response in the format of a QA dataset. The human input should not be a one-word answer question like "What is the capital of France?". The AI response generated should be a few sentences long."""
GENERATION_PROMPT_WITH_GROUND_TRUTH = """Please generate:
1. A random human message (not a simple one-word answer question)
2. An AI response (a few sentences long)
3. A perfect reference answer that would score 5/5 on all criteria (e.g., concise, helpful, and accurate)

Format as JSON with "human", "ai", and "ground_truth" fields."""

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
    
    # Randomly choose between GPT-3.5 and Claude with 65%/35% weights
    model_choice = random.choices([
        ("gpt-3.5-turbo", get_openai_response),
        ("claude-3-5-haiku-latest", get_anthropic_response)
    ], weights=[0.5, 0.5])[0]
    model_name, api_func = model_choice
    
    # Generate response using selected model
    response = api_func(
        model_name=model_name,
        prompt=GENERATION_PROMPT,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=1
    )
    
    # Define default messages
    default_human = "How do muscles grow?"
    default_ai = """Muscles grow through a process called skeletal muscle hypertrophy, which adds more myosin filaments to each muscle fiber, making the engine of the cell bigger and stronger over time. This is achieved through increased muscle tension and physical stress, breaking down muscle fiber. Muscle growth is also a direct consequence of resistance training and nutrition. People build muscle at different rates depending on their age, sex, and genetics, but muscle development significantly increases if exercise is done correctly and the body stores more protein through a process called protein synthesis."""
    
    try:
        # Clean the response by replacing newlines with spaces
        cleaned_response = response.replace('\n', ' ').replace('\r', '')
        data = json.loads(cleaned_response)
        
        # Extract messages with fallbacks
        human_message = data.get("human", default_human)
        ai_message = data.get("ai", default_ai)
        
        # Debug logging
        print(f"Parsed response: human='{human_message}', ai='{ai_message[:50]}...'")
        
    except Exception as e:
        print(f"Failed to parse response: {str(e)}\n {response}")
        human_message = default_human
        ai_message = default_ai
    
    return human_message, ai_message

def get_random_human_ai_ground_truth_pair():
    # Select system prompt with specified probabilities
    system_prompts = {
        "good": GOOD_SYSTEM_PROMPT_WITH_GROUND_TRUTH,
        "bad": BAD_SYSTEM_PROMPT_WITH_GROUND_TRUTH,
        "ambiguous": AMBIGUOUS_SYSTEM_PROMPT_WITH_GROUND_TRUTH
    }
    
    prompt_type = random.choices(
        ["good", "bad", "ambiguous"],
        weights=[0.2, 0.2, 0.6]  # 20% good, 20% bad, 60% ambiguous
    )[0]
    
    system_prompt = system_prompts[prompt_type]
    print(f"Generating {prompt_type} response with ground truth")
    
    # Randomly choose between GPT-3.5 and Claude with 50/50 weights
    model_choice = random.choices([
        ("gpt-3.5-turbo", get_openai_response),
        ("claude-3-5-haiku-latest", get_anthropic_response)
    ], weights=[0.5, 0.5])[0]
    model_name, api_func = model_choice
    
    # Define default messages
    defaults = {
        "human": "How do muscles grow?",
        "ai": """Muscles grow through a process called skeletal muscle hypertrophy, which adds more myosin filaments to each muscle fiber, making the engine of the cell bigger and stronger over time. This is achieved through increased muscle tension and physical stress, breaking down muscle fiber. Muscle growth is also a direct consequence of resistance training and nutrition. People build muscle at different rates depending on their age, sex, and genetics, but muscle development significantly increases if exercise is done correctly and the body stores more protein through a process called protein synthesis.""",
        "ground_truth": """Muscle growth (hypertrophy) occurs through a complex biological process involving several key mechanisms:

1. Mechanical Tension: Resistance training creates mechanical tension in muscle fibers, triggering molecular and cellular responses that promote growth.

2. Metabolic Stress: The depletion of energy resources and accumulation of metabolic byproducts during exercise contributes to muscle growth signaling.

3. Muscle Damage: Exercise-induced micro-damage to muscle fibers activates satellite cells, which help repair and build new muscle tissue.

4. Protein Synthesis: After exercise, increased protein synthesis rates exceed protein breakdown, leading to net muscle protein accretion.

5. Hormonal Response: Exercise triggers the release of growth-promoting hormones like testosterone, growth hormone, and IGF-1.

6. Recovery: Adequate rest between training sessions allows for repair and growth, supported by proper nutrition, particularly protein intake (1.6-2.2g/kg/day).

This process is influenced by factors including genetics, age, sex, nutrition, sleep quality, and training variables. Optimal muscle growth requires a structured resistance training program, adequate protein intake, sufficient calories, and proper recovery."""
    }
    
    # Generate response using selected model
    response = api_func(
        model_name=model_name,
        prompt=GENERATION_PROMPT_WITH_GROUND_TRUTH,
        system_prompt=system_prompt,
        max_tokens=1000,  # Increased token limit to accommodate ground truth
        temperature=1
    )
    
    # Parse the response to get all three components
    try:
        # Clean the response by replacing newlines with spaces
        cleaned_response = response.replace('\n', ' ').replace('\r', '')
        data = json.loads(cleaned_response)
        
        # Extract messages with fallbacks
        human_message = data.get("human", defaults["human"])
        ai_message = data.get("ai", defaults["ai"])
        ground_truth = data.get("ground_truth", defaults["ground_truth"])
        
        # Debug logging
        print(f"Parsed response: human='{human_message}', ai='{ai_message[:50]}...', ground_truth='{ground_truth[:50]}...'")
        
    except Exception as e:
        print(f"Failed to parse response: {str(e)}\n {response}")
        human_message = defaults["human"]
        ai_message = defaults["ai"]
        ground_truth = defaults["ground_truth"]
    
    return human_message, ai_message, ground_truth

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