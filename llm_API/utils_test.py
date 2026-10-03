from utils import (
    generate_with_single_input, # (prompt = query, role = 'user')
    generate_with_multiple_input,
    get_proxy_url,
    get_proxy_headers, 
    get_together_key
)

# Example call
output = generate_with_single_input(
    prompt="What is the capital of France?"
)

print("Role:", output['role'])
print("Content:", output['content'])