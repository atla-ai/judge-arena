from openai import OpenAI
import anthropic
from together import Together
import cohere
import json
import re
import os
import requests
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    PROMETHEUS_PROMPT,
    PROMETHEUS_PROMPT_WITH_REFERENCE,
    ATLA_PROMPT,
    ATLA_PROMPT_WITH_REFERENCE,
     FLOW_JUDGE_PROMPT
)
from transformers import AutoTokenizer
from atla import Atla

# Initialize clients
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
together_client = Together()
hf_api_key = os.getenv("HF_API_KEY")
flow_judge_api_key = os.getenv("FLOW_JUDGE_API_KEY")
cohere_client = cohere.ClientV2(os.getenv("CO_API_KEY"))
salesforce_api_key = os.getenv("SALESFORCE_API_KEY")

# Initialize Atla client
atla_client = Atla()

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

def get_prometheus_response(model_name, prompt, system_prompt=None, max_tokens=500, temperature=0.01):
    """Get response from Hugging Face model"""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create messages list for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        model_id = "prometheus-eval/prometheus-7b-v2.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "temperature": temperature
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

def get_atla_response(model_name, prompt, system_prompt=None, max_tokens=500, temperature=0.01):
    """Get response from Atla API"""
    try:
        # Extract components from the prompt data
        model_input = prompt.get('human_input', '')
        model_output = prompt.get('ai_response', '')
        expected_output = prompt.get('ground_truth_input', '')
        evaluation_criteria = prompt.get('eval_criteria', '')

        # Set model_id based on the model name
        if "Mini" in model_name:
            model_id = "atla-selene-mini"
        else:
            model_id = "atla-selene"

        response = atla_client.evaluation.create(
            model_id=model_id,
            model_input=model_input,
            model_output=model_output,
            expected_model_output=expected_output if expected_output else None,
            evaluation_criteria=evaluation_criteria,
        )
        
        # Return the score and critique directly
        return {
            "score": response.result.evaluation.score,
            "critique": response.result.evaluation.critique
        }
    except Exception as e:
        return f"Error with Atla model {model_name}: {str(e)}"

def get_flow_judge_response(model_name, prompt, max_tokens=2048, temperature=0.1, top_p=0.95) -> str:
    """Get response from Flow Judge"""
    try:
        response = requests.post(
            "https://arena.flow-ai.io/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {flow_judge_api_key}"
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": None
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]['message']['content']
    except Exception as e:
        return f"Error with Flow Judge completions model {model_name}: {str(e)}"

def get_cohere_response(model_name, prompt, system_prompt=JUDGE_SYSTEM_PROMPT, max_tokens=500, temperature=0):
    """Get response from Cohere API"""
    try:
        response = cohere_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Extract the text from the content items
        content_items = response.message.content
        if isinstance(content_items, list):
            # Get the text from the first content item
            return content_items[0].text
        return str(content_items)  # Fallback if it's not a list
    except Exception as e:
        return f"Error with Cohere model {model_name}: {str(e)}"

def get_salesforce_response(model_name, prompt, system_prompt=None, max_tokens=2048, temperature=0):
    """Get response from Salesforce Research API"""
    try:
        headers = {
            'accept': 'application/json',
            "content-type": "application/json",
            "X-Api-Key": salesforce_api_key,
        }

        # Create messages list
        messages = []
        messages.append({"role": "user", "content": prompt})

        json_data = {
            "prompts": messages,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            'https://gateway.salesforceresearch.ai/sfr-judge/process',
            headers=headers,
            json=json_data
        )
        response.raise_for_status()
        return response.json()['result'][0]
    except Exception as e:
        return f"Error with Salesforce model {model_name}: {str(e)}"

def get_model_response(
    model_name,
    model_info,
    prompt_data,
    use_reference=False,
    max_tokens=500,
    temperature=0
):
    """Get response from appropriate API based on model organization"""
    if not model_info:
        return "Model not found or unsupported."

    api_model = model_info["api_model"]
    organization = model_info["organization"]

    # Determine if model is Prometheus, Atla, Flow Judge, or Salesforce
    is_prometheus = (organization == "Prometheus")
    is_atla = (organization == "Atla")
    is_flow_judge = (organization == "Flow AI")
    is_salesforce = (organization == "Salesforce")  
    
    # For non-Prometheus/Atla/Flow Judge/Salesforce models, use the Judge system prompt
    system_prompt = None if (is_prometheus or is_atla or is_flow_judge or is_salesforce) else JUDGE_SYSTEM_PROMPT

    # Select the appropriate base prompt
    if is_atla or is_salesforce:  # Use same prompt for Atla and Salesforce
        base_prompt = ATLA_PROMPT_WITH_REFERENCE if use_reference else ATLA_PROMPT
    elif is_flow_judge:
        base_prompt = FLOW_JUDGE_PROMPT
    else:
        base_prompt = PROMETHEUS_PROMPT_WITH_REFERENCE if use_reference else PROMETHEUS_PROMPT
    
    # For non-Prometheus/non-Atla/non-Salesforce models, use Prometheus but replace the output format with JSON
    if not (is_prometheus or is_atla or is_flow_judge or is_salesforce):
        base_prompt = base_prompt.replace(
            '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"',
            '3. Your output format should strictly adhere to JSON as follows: {{"feedback": "<write feedback>", "result": <numerical score>}}. Ensure the output is valid JSON, without additional formatting or explanations.'
        )

    try:
        if not is_flow_judge:
            # Format the prompt with the provided data, only using available keys
            final_prompt = base_prompt.format(
                human_input=prompt_data['human_input'],
                ai_response=prompt_data['ai_response'],
                ground_truth_input=prompt_data.get('ground_truth_input', ''),
                eval_criteria=prompt_data['eval_criteria'],
                score1_desc=prompt_data['score1_desc'],
                score2_desc=prompt_data['score2_desc'],
                score3_desc=prompt_data['score3_desc'],
                score4_desc=prompt_data['score4_desc'],
                score5_desc=prompt_data['score5_desc']
            )
        else:
            human_input = f"<user_input>\n{prompt_data['human_input']}\n</user_input>"
            ai_response = f"<response>\n{prompt_data['ai_response']}\n</response>"
            ground_truth=prompt_data.get('ground_truth_input', '')
            if ground_truth:
                response_reference = f"<response_reference>\n{ground_truth}\n</response_reference>"
            else:
                response_reference = ""
            eval_criteria = prompt_data['eval_criteria']
            score1_desc = f"- Score 1: {prompt_data['score1_desc']}\n"
            score2_desc = f"- Score 2: {prompt_data['score2_desc']}\n"
            score3_desc = f"- Score 3: {prompt_data['score3_desc']}\n"
            score4_desc = f"- Score 4: {prompt_data['score4_desc']}\n"
            score5_desc = f"- Score 5: {prompt_data['score5_desc']}"
            rubric = score1_desc + score2_desc + score3_desc + score4_desc + score5_desc
            if response_reference:
                inputs = human_input + "\n"+ response_reference
            else:
                inputs = human_input
            final_prompt = base_prompt.format(
                INPUTS=inputs,
                OUTPUT=ai_response,
                EVALUATION_CRITERIA=eval_criteria,
                RUBRIC=rubric
            )
        
    except KeyError as e:
        return f"Error formatting prompt: Missing required field {str(e)}"

    try:
        if organization == "OpenAI":
            return get_openai_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Anthropic":
            return get_anthropic_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Prometheus":
            return get_prometheus_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature = 0.01
            )
        elif organization == "Atla":
            response = get_atla_response(
                api_model, prompt_data, system_prompt, max_tokens, temperature
            )
            # Response now contains score and critique directly
            if isinstance(response, dict) and 'score' in response and 'critique' in response:
                score = str(response['score'])
                critique = response['critique']
                return score, critique
            else:
                return "Error", str(response)
        elif organization == "Cohere":
            return get_cohere_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
        elif organization == "Flow AI":
            return get_flow_judge_response(
                api_model, final_prompt
            )
        elif organization == "Salesforce":
            response = get_salesforce_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )           
            return response
        else:
            # All other organizations use Together API
            return get_together_response(
                api_model, final_prompt, system_prompt, max_tokens, temperature
            )
    except Exception as e:
        return f"Error with {organization} model {model_name}: {str(e)}"

def parse_model_response(response):
    try:
        # Debug print
        print(f"Raw model response: {response}")

        # If response is already a tuple (from Atla/Salesforce), use it directly
        if isinstance(response, tuple):
            return response

        # If response is already a dictionary, use it directly
        if isinstance(response, dict):
            return str(response.get("result", "N/A")), response.get("feedback", "N/A")

        # First try to parse the entire response as JSON
        try:
            data = json.loads(response)
            return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except json.JSONDecodeError:
            # If that fails, check if this is a Salesforce response
            if "**Reasoning:**" in response or "**Result:**" in response:
                # Use ATLA parser for Salesforce responses only
                return salesforce_parse_model_response(response)
                
            # Otherwise try to find JSON within the response
            json_match = re.search(r"{.*}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
            else:
                return "Error", f"Invalid response format returned - here is the raw model response: {response}"

    except Exception as e:
        # Debug print for error case
        print(f"Failed to parse response: {str(e)}")
        
        # If the error message itself contains valid JSON, try to parse that
        try:
            error_json_match = re.search(r"{.*}", str(e), re.DOTALL)
            if error_json_match:
                data = json.loads(error_json_match.group(0))
                return str(data.get("result", "N/A")), data.get("feedback", "N/A")
        except:
            pass
            
        return "Error", f"Failed to parse response: {response}"
    
def prometheus_parse_model_response(output):
    try:
        print(f"Raw model response: {output}")
        output = output.strip()

        # Remove "Feedback:" prefix if present (case insensitive)
        output = re.sub(r'^feedback:\s*', '', output, flags=re.IGNORECASE)
        
        # New pattern to match [RESULT] X at the beginning
        begin_result_pattern = r'^\[RESULT\]\s*(\d+)\s*\n*(.*?)$'
        begin_match = re.search(begin_result_pattern, output, re.DOTALL | re.IGNORECASE)
        if begin_match:
            score = int(begin_match.group(1))
            feedback = begin_match.group(2).strip()
            return str(score), feedback

        # Existing patterns for end-of-string results...
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

def salesforce_parse_model_response(output):
    """Parse response from Salesforce model"""
    try:
        print(f"Raw Salesforce model response: {output}")
        output = output.strip()
        
        # Look for the Reasoning and Result sections
        reasoning_match = re.search(r'\*\*Reasoning:\*\*(.*?)(?=\*\*Result:|$)', output, re.DOTALL)
        result_match = re.search(r'\*\*Result:\*\*\s*(\d+)', output)
        
        if reasoning_match and result_match:
            feedback = reasoning_match.group(1).strip()
            score = result_match.group(1)
            return str(score), feedback
            
        return "Error", f"Failed to parse Salesforce response format: {output}"

    except Exception as e:
        print(f"Failed to parse Salesforce response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"
    
def flow_judge_parse_model_response(output):
    try:
        print(f"Raw model response: {output}")
        # Convert multiple line breaks to single ones and strip whitespace
        output = re.sub(r'\n{2,}', '\n', output.strip())
        
        # Compile regex patterns
        feedback_pattern = re.compile(r"<feedback>\s*(.*?)\s*</feedback>", re.DOTALL)
        score_pattern = re.compile(r"<score>\s*(\d+)\s*</score>", re.DOTALL)

        feedback_match = feedback_pattern.search(output)
        score_match = score_pattern.search(output)

        if feedback_match or not score_match:
            feedback = feedback_match.group(1).strip()
            score = int(score_match.group(1).strip())
            return str(score), feedback
            
        return "Error", f"Failed to parse response: {output}"
        
    except Exception as e:
        print(f"Failed to parse response: {str(e)}")
        return "Error", f"Exception during parsing: {str(e)}"