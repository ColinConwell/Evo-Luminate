import json

import aisuite as ai

llm_client = ai.Client()

generation_prompt = (
    "Generate a random JSON object with the following keys: 'name', 'age', 'email'."
)

response = llm_client.chat.completions.create(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": generation_prompt}],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)

# print(response.choices[0].message)

# try:
#     idea_json = json.loads(response.choices[0].message.content)
#     print(idea_json)
# except json.JSONDecodeError:
#     print("Failed to decode JSON from the response.")
