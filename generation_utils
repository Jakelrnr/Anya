def generate_response(model, message, personality_settings=None):
    """
    Generate AI response with detailed parameter explanations
    """
    
    # Default personality settings
    if personality_settings is None:
        personality_settings = {
            "temperature": 0.8,    # Creativity (0.1=boring, 1.5=chaotic)
            "top_p": 0.9,         # Word choice diversity  
            "top_k": 40,          # Limits word choices
            "repeat_penalty": 1.1, # Prevents repetition
            "max_tokens": 200     # Response length limit
        }
    
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a friendly, energetic AI assistant."},
            {"role": "user", "content": message}
        ],
        
        # Core generation parameters
        temperature=personality_settings["temperature"],
        top_p=personality_settings["top_p"],
        top_k=personality_settings["top_k"],
        repeat_penalty=personality_settings["repeat_penalty"],
        max_tokens=personality_settings["max_tokens"],
        
        # Advanced settings
        stream=False,           # Set True for word-by-word streaming
        stop=["Human:", "\n\n"], # Stop generation at these tokens
    )
    
    return response['choices'][0]['message']['content']

# Example: Different personalities through parameters
energetic_settings = {
    "temperature": 0.9,      # High creativity
    "top_p": 0.95,          # More word variety
    "repeat_penalty": 1.2   # Avoid repetition
}

calm_settings = {
    "temperature": 0.6,      # Lower creativity
    "top_p": 0.8,           # More focused responses
    "repeat_penalty": 1.05  # Less repetition penalty,
}
