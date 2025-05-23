import os
from dotenv import load_dotenv
from gen_api_answer import get_model_response, atla_parse_model_response

# Load environment variables
load_dotenv()

def test_salesforce_model_response():
    # Test data
    model_name = "SFR-LLaMA-3.1-70B-Judge"
    model_info = {
        "organization": "Salesforce",
        "api_model": "sfr-llama-3.1-70b-judge"
    }
    
    # Update prompt_data to match the format expected by the application
    prompt_data = {
        'human_input': "What is 2+2?",
        'ai_response': "The sum of 2 and 2 is 4.",
        'eval_criteria': "Mathematical accuracy and clarity of explanation",
        'score1_desc': "Completely incorrect answer with poor explanation",
        'score2_desc': "Partially correct answer or unclear explanation",
        'score3_desc': "Correct answer but basic explanation",
        'score4_desc': "Correct answer with clear explanation",
        'score5_desc': "Correct answer with excellent, detailed explanation",
        'ground_truth_input': None  # Add this to match the expected format
    }
    
    print("Testing get_model_response with Salesforce...")
    print(f"Prompt data: {prompt_data}")
    
    try:
        response = get_model_response(
            model_name=model_name,
            model_info=model_info,
            prompt_data=prompt_data
        )
        
        print("\nResponse:")
        print(response)
        
        # Add parsing test
        score, feedback = atla_parse_model_response(response)
        print("\nParsed Response:")
        print(f"Score: {score}")
        print(f"Feedback: {feedback}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    test_salesforce_model_response()