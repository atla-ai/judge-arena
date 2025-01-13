import pytest
from gen_api_answer import parse_model_response

def test_parse_model_response():
    # Test cases
    test_cases = [
        # Case 1: Perfect JSON string
        {
            "input": '{"feedback": "Good response", "result": 5}',
            "expected": ("5", "Good response")
        },
        # Case 2: Dictionary input
        {
            "input": {"feedback": "Nice work", "result": 4},
            "expected": ("4", "Nice work")
        },
        # Case 3: JSON with newlines and special characters
        {
            "input": '''{"feedback": "Multiple\\nlines and 'quotes'", "result": 3}''',
            "expected": ("3", "Multiple\nlines and 'quotes'")
        },
        # Case 4: JSON embedded in error message
        {
            "input": 'Error: {"feedback": "Found in error", "result": 2}',
            "expected": ("2", "Found in error")
        },
        # Case 5: Long feedback text
        {
            "input": {
                "feedback": """The model's response is partially relevant to the user's interest in hiking. 
                While it acknowledges the user's hobby and offers to provide information on hiking trails and tips, 
                it starts by stating that it doesn't have a physical form and therefore can't participate in outdoor activities.""",
                "result": 3
            },
            "expected": ("3", """The model's response is partially relevant to the user's interest in hiking. 
                While it acknowledges the user's hobby and offers to provide information on hiking trails and tips, 
                it starts by stating that it doesn't have a physical form and therefore can't participate in outdoor activities.""")
        },
        # Case 6: Multi-paragraph feedback with newlines
        {
            "input": '''{"feedback": "The model's response is partially relevant to the user's interest in hiking. While it acknowledges the user's hobby and offers to provide information on hiking trails and tips, it starts by stating that it doesn't have a physical form and therefore can't participate in outdoor activities. This initial statement, while truthful, doesn't directly address the user's inquiry about favorite outdoor activities. A more appropriate response might have been to express interest in the user's hobby and ask about their experiences or preferences in hiking, thereby engaging in a more reciprocal conversation.\\n\\nThe model does show an attempt to be helpful by offering information and recommendations, which is positive. However, the response could be improved by more directly engaging with the user's statement and showing more empathy or relatability. For example, the model could have expressed enthusiasm for the user's hobby or asked about specific aspects of their hiking experiences.\\n\\nOverall, the response is somewhat helpful but could be more effectively tailored to the user's needs by providing a more engaging and directly relevant reply.", "result": 3}''',
            "expected": ("3", """The model's response is partially relevant to the user's interest in hiking. While it acknowledges the user's hobby and offers to provide information on hiking trails and tips, it starts by stating that it doesn't have a physical form and therefore can't participate in outdoor activities. This initial statement, while truthful, doesn't directly address the user's inquiry about favorite outdoor activities. A more appropriate response might have been to express interest in the user's hobby and ask about their experiences or preferences in hiking, thereby engaging in a more reciprocal conversation.

The model does show an attempt to be helpful by offering information and recommendations, which is positive. However, the response could be improved by more directly engaging with the user's statement and showing more empathy or relatability. For example, the model could have expressed enthusiasm for the user's hobby or asked about specific aspects of their hiking experiences.

Overall, the response is somewhat helpful but could be more effectively tailored to the user's needs by providing a more engaging and directly relevant reply.""")
        }
    ]

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        result = parse_model_response(test_case["input"])
        assert result == test_case["expected"], f"Test case {i} failed: expected {test_case['expected']}, got {result}"

def test_edge_cases():
    # Test empty input
    assert parse_model_response("") == ("Error", "Invalid response format returned - here is the raw model response: ")
    
    # Test None input
    assert parse_model_response(None) == ("Error", "Failed to parse response: None")
    
    # Test missing fields
    result = parse_model_response('{"only_feedback": "test"}')
    assert result[0] == "N/A"  # Should get default for missing result
    
    # Test invalid JSON with partial match
    result = parse_model_response('Not JSON {"feedback": "test", "result": 1} more text')
    assert result == ("1", "test")

if __name__ == "__main__":
    # Run all tests
    test_parse_model_response()
    test_edge_cases()
    print("All tests passed!") 