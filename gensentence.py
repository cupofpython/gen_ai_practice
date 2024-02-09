from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Define starter text
text = "Sam is"

# Generate text until a sentence is complete.
while  not (".") in text and len(text) < 100:
    # Get tokens
    inputs = tokenizer(text, return_tensors="pt")
    inputs["input_ids"]     
    # Generate probabilities
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probabilities = torch.nn.functional.softmax(logits[0], dim=-1)
    
    # Get list of tuples with ids and probabilities
    probabilities = [(id, p.item()) for id, p in enumerate(probabilities)]

    # Get next word based on top probability in list
    next_word = sorted(probabilities, key=lambda x: x[1], reverse=True)[0]

    # Append text
    text = text + tokenizer.decode(next_word[0])
    print(text)

print("Final result: " + text)
