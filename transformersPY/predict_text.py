import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding
from sklearn.base import BaseEstimator, ClassifierMixin

### Custom estimator for transformer model
class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, max_length=512, epochs=5, batch_size=1):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        self.model = None

    def _tokenize(self, texts):
        return self.tokenizer(
            texts.astype(str).tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="tf"
        )
    
    def predict_text(self, text):
        if self.model is None:
            print("Loading best saved validated model...")
            self.model = tf.keras.models.load_model('distilbert-base-uncased_original')

        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="tf"
        )
        output = self.model(tokenized)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output.logits
        prediction = np.argmax(logits, axis=1)[0]
        return prediction
    
if __name__ == "__main__":
    ### Deduce real email text body  ||| Replace text in single_text to the email body
    label_mapping = {0: "Safe Email", 1: "Phishing Email"}  #Is set to a 0/1 binary system for labeling safe and phishing
    real_email = TransformerClassifier(batch_size=1)        #Initialize class and set batch_size to 1 since it is only one text is being run
    single_text = """ REPLACE TEXT HERE """
    pred_single = real_email.predict_text(single_text)
    print(f"\nDirect prediction for text: \n'{single_text}' \n\n --> Prediction: {label_mapping[pred_single]}")

    