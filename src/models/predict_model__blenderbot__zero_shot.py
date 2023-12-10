#https://huggingface.co/facebook/bart-large-mnli
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


customer_texts = [
    "I want to find out more about your product",
    "I need to speak to someone your product has broken and i want you to do something about it",
    "fuck you, you useless piece of shit"
]


candidate_labels = ['angry', 'not angry']


for text_to_classify in customer_texts:
    print("")
    print(text_to_classify)
    response = classifier(text_to_classify, candidate_labels)
    for i, e in enumerate(response["labels"]):
        print(f"{e} = {response['scores'][i]}")
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}

