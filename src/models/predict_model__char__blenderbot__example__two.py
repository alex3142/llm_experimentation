#https://huggingface.co/docs/transformers/model_doc/blenderbot#usage
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

mname = "facebook/blenderbot-400M-distill"

model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
UTTERANCE = "My friends are cool but they eat too many carbs."

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)

reply_ids = model.generate(**inputs)

print(tokenizer.batch_decode(reply_ids))