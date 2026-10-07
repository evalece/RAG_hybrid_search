from utils import (
    generate_with_single_input, # (prompt = query, role = 'user')

)

# Example call
output = generate_with_single_input(
    prompt="What is the capital of France?"
)

print("Role:", output['role'])
print("Content:", output['content'])