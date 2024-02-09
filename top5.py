from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Define starter text
text = "Sam is"

# Get tokens
inputs = tokenizer(text, return_tensors="pt")
inputs["input_ids"]

# Get logits
with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)


# Built in method for top 5
#top_5 = torch.topk(probabilities, 5)
#print([tokenizer.decode(id) for id in top_5.indices])


# Get top 5 probabilities and their ids

probabilities = [(id, p.item()) for id, p in enumerate(probabilities)]

top_5 = sorted(probabilities, key=lambda x: x[1], reverse=True)[:5]
for id, p in top_5:
    print(tokenizer.decode(id))