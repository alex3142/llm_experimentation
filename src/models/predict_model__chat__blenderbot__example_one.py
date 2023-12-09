#https://huggingface.co/docs/transformers/model_doc/blenderbot
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)

tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
