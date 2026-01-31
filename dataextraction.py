import polars as pl
import json
#import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# Initialize the OpenAI client with NVIDIA API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-BS6d9r3E" #API key
)
# Read the CSV file using Polars
df = pl.read_csv("/content/AI_Human.csv", encoding="ISO-8859-1", ignore_errors=True)
# Filter the rows where the 'generated' column equals 0 which is human input
human = df.filter(pl.col('generated') == 0)
# Convert to a list of input texts
inputs = human['text'].to_list()
# Initialize the result list
result = []
# List of revision prompts
prompts = [
    "Revise this with your best effort",
    "Help me polish this",
    "Rewrite this for me",
    "Make this fluent while doing minimal change",
    "Refine this for me please",
    "Concise this for me and keep all the information"
]
inputs_to_process = inputs[:1000]

def process_input(args):
    input_text, index = args
    input_dict = {"input": input_text}
    for prompt in prompts:
        # Construct the full prompt
        full_prompt = f"{prompt}:\n\n{input_text}"
        # Create a chat completion request
        completion = client.chat.completions.create(
            extra_body={},
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )
        # Extract the revised text from the response
        if completion.choices and len(completion.choices) > 0:
            revised_text = completion.choices[0].message.content
        else:
            revised_text = "No response from API"
        input_dict[prompt] = revised_text
    print(f"Processed input {index + 1} of {len(inputs_to_process)}")
    return input_dict
def process_and_print(input_text, index):
    result = process_input((input_text, index))
    print(f"Finished processing input {index + 1}")
    return result
# Using ThreadPoolExecutor to parallelize the processing
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_and_print, inputs_to_process, range(len(inputs_to_process))))
# Save the results to a JSON file
with open("human1-1000.json", "w") as f:
    json.dump(results, f, indent=4)
print("Processing complete. Results saved to gpt364-366.json.")
