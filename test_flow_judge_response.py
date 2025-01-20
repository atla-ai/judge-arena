import os
from dotenv import load_dotenv
from gen_api_answer import get_model_response, flow_judge_parse_model_response

# Load environment variables
load_dotenv()

def test_flow_judge_model_response():
    # Test data
    model_name = "flow-judge-v1"
    model_info = {
        "organization": "Flow AI",
        "api_model": "flow-judge-v1"
    }
    
    # Test prompt data
    prompt_data = {
        'human_input': "What is 2+2?",
        'ai_response': "The sum of 2 and 2 is 4.",
        'eval_criteria': "Mathematical accuracy and clarity of explanation",
        'score1_desc': "Completely incorrect answer with poor explanation",
        'score2_desc': "Partially correct answer or unclear explanation",
        'score3_desc': "Correct answer but basic explanation",
        'score4_desc': "Correct answer with clear explanation",
        'score5_desc': "Correct answer with excellent, detailed explanation",
        'ground_truth_input': None
    }
    
    print("Testing get_model_response with Flow Judge...")
    print(f"Prompt data: {prompt_data}")
    
    try:
        response = get_model_response(
            model_name=model_name,
            model_info=model_info,
            prompt_data=prompt_data,
            max_tokens=500,
            temperature=0.1
        )
        
        print("\nResponse:")
        print(response)
        
        # Add parsing test
        score, feedback = flow_judge_parse_model_response(response)
        print("\nParsed Response:")
        print(f"Score: {score}")
        print(f"Feedback: {feedback}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    test_flow_judge_model_response() 