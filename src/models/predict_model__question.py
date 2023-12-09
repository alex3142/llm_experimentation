from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
import logging

model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

logging.info("loaded models...")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's the capital of France?"
context = "You  are a friendly chatbot"
pred = onnx_qa(question, context)