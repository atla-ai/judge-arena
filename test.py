from gen_api_answer import get_model_response, alternative_parse_model_response


# Test model configuration
model_info = {
    "name": "Command-R Plus",
    "organization": "Cohere",
    "api_model": "command-r-plus"
}

# Sample data
test_input = "Write a haiku about autumn"
test_response = """Autumn leaves falling
Gently to the forest floor
Nature's soft goodbye"""
test_reference = """Crimson maple leaves
Dance on crisp October winds
Time slips through branches"""

# Construct prompt in compatible mode format
test_prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other openings, closings, or explanations.

###The instruction to evaluate:
{test_input}

###Response to evaluate:
{test_response}

###Reference Answer (Score 5):
{test_reference}

###Score Rubrics:
[Evaluate the haiku based on adherence to form (5-7-5 syllable pattern), imagery, and seasonal reference]
Score 1: Fails to follow haiku structure and lacks both imagery and seasonal elements
Score 2: Partially follows structure but lacks either imagery or seasonal elements
Score 3: Follows structure and includes basic imagery or seasonal elements
Score 4: Well-structured with good imagery and clear seasonal reference
Score 5: Perfect structure with exceptional imagery and profound seasonal insight

###Feedback:"""

def test_model():
    print("Testing Command-R Plus with compatible mode...\n")
    
    # Get response
    response = get_model_response(
        model_name=model_info['name'],
        model_info=model_info,
        prompt=test_prompt,
        use_alternative_prompt=True,
        max_tokens=500,
        temperature=0
    )
    
    print("Raw response:")
    print(response)
    print("\nParsing response...")
    
    # Parse response
    score, feedback = alternative_parse_model_response(response)
    
    print("\nParsed results:")
    print(f"Score: {score}")
    print(f"Feedback: {feedback}")

# Run test
if __name__ == "__main__":
    test_model()