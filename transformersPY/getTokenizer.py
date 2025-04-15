from transformers import DistilBertTokenizerFast 
import tensorflow as tf

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("./tokenizer")

