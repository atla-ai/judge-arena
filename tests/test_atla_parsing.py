def test_get_model_response_atla():
    # Sample judge arena input
    judge_input = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Are you sure?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Mock the expected Atla response
    expected_response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Yes, I am absolutely sure that 2+2=4. This is one of the most basic mathematical facts."
            }
        }]
    }
    
    # You'll need to mock the actual API call here
    with patch('your_module.atla_client.complete', return_value=expected_response):
        response = get_model_response(judge_input)
        
        # Assert the response matches expected format
        assert isinstance(response, dict)
        assert "content" in response
        assert response["content"] == expected_response["choices"][0]["message"]["content"]
        assert "role" in response
        assert response["role"] == "assistant"
