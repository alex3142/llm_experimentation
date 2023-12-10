#https://huggingface.co/docs/transformers/model_doc/blenderbot#usage
from transformers import (
   BlenderbotTokenizer,
   BlenderbotForConditionalGeneration,
   pipeline,
)

model_name = "facebook/blenderbot-400M-distill"

model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

messages = [
    {
        "role": "system",
        "content": "Your job is to tell me who is most likely to complain.",
    },
    {
       "role": "user", "content": "Alice said 'fuck you'?, Bob said: 'I'd like to speak to a human please'"
    },
]

pipe = pipeline(
    "conversational",
    model=model,
    tokenizer=tokenizer,
)

print(pipe(messages))
