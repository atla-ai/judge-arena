from openai import OpenAI
import anthropic
from together import Together
import json
import re
import os
import requests

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()
hf_api_key = os.getenv("HF_API_KEY")
huggingface_client = OpenAI(
    base_url="https://otb7jglxy6r37af6.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    api_key=hf_api_key
)

JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction. Your output format should strictly adhere to JSON as follows: {"feedback": "<write feedback>", "result": <numerical score>}. Ensure the output is valid JSON, without additional formatting or explanations."""

ALTERNATIVE_JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate based on the user's instruction."""

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

def get_hf_response(model_name, prompt, max_tokens=500):
    """Get response from Hugging Face model"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            "https://otb7jglxy6r37af6.us-east-1.aws.endpoints.huggingface.cloud",
            headers=headers,
            json=payload
        )
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"Error with Hugging Face model {model_name}: {str(e)}"

def get_model_response(
    model_name,
    model_info,
    prompt,
    use_alternative_prompt=False,
    max_tokens=500,
    temperature=0
):
    """Get response from appropriate API based on model organization"""
    if not model_info:
        return "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    # Select the appropriate system prompt
    if use_alternative_prompt:
        system_prompt = ALTERNATIVE_JUDGE_SYSTEM_PROMPT
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT

    try:
        if organization == "OpenAI":
            return get_openai_response(
                api_model, prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Anthropic":
            return get_anthropic_response(
                api_model, prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Prometheus":
            return get_hf_response(
                api_model, prompt, max_tokens
            )
        else:
            # All other organizations use Together API
            return get_together_response(
                api_model, prompt, system_prompt, max_tokens, temperature
            )
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
    
def alternative_parse_model_response(output):
    try:
        print(f"Raw model response: {output}")

        # Remove "Feedback:" prefix if present (case insensitive)
        output = re.sub(r'^feedback:\s*', '', output.strip(), flags=re.IGNORECASE)

        # First, try to match the pattern "... [RESULT] X"
        pattern = r"(.*?)\s*\[RESULT\]\s*[\(\[]?(\d+)[\)\]]?"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # If no match, try to match "... Score: X"
        pattern = r"(.*?)\s*(?:Score|Result)\s*:\s*[\(\[]?(\d+)[\)\]]?"
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # Pattern to handle [Score X] at the end
        pattern = r"(.*?)\s*\[(?:Score|Result)\s*[\(\[]?(\d+)[\)\]]?\]$"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            feedback = match.group(1).strip()
            score = int(match.group(2))
            return str(score), feedback

        # Final fallback attempt
        pattern = r"[\(\[]?(\d+)[\)\]]?\s*\]?$"
        match = re.search(pattern, output)
        if match:
            score = int(match.group(1))
            feedback = output[:match.start()].rstrip()
            # Remove any trailing brackets from feedback
            feedback = re.sub(r'\s*\[[^\]]*$', '', feedback).strip()
            return str(score), feedback

        return "Error", f"Failed to parse response: {output}"

    except Exception as e:
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"