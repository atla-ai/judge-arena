import requests
import json
from gen_api_answer import salesforce_api_key

def test_salesforce_api():
    headers = {
        'accept': 'application/json',
        "content-type": "application/json",
        "X-Api-Key": salesforce_api_key,
    }

    # Simple test input
    test_input = "Please evaluate this response. Human: What is 2+2? Assistant: The sum of 2 and 2 is 4."

    json_data = {
        "prompts": [
            {
                "role": "user",
                "content": test_input,
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 2048,
    }

    try:
        print("Sending request to Salesforce API...")
        print(f"Input: {test_input}")
        
        response = requests.post(
            'https://gateway.salesforceresearch.ai/sfr-judge/process',
            headers=headers,
            json=json_data
        )
        
        print("\nResponse status:", response.status_code)
        print("Response headers:", dict(response.headers))
        
        if response.status_code == 200:
            result = response.json()['result'][0]
            print("\nSuccess! Response:")
            print(result)
        else:
            print("\nError Response:")
            print(response.text)
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    test_salesforce_api() 