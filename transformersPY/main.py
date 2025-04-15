import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

### Custom estimator for transformer model
class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, max_length=512, epochs=5, batch_size=64):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)    
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

    def fit(self, X, y):
        # Tokenize the texts & convert labels to TensorFlow tensor
        tokenized = self._tokenize(X)
        y_tensor = tf.convert_to_tensor(y)

        # Create the model
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )

        # Prepare the tensorflow dataset $ shuffle and batch the dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized), y_tensor))
        dataset = dataset.shuffle(len(X), seed=seed).batch(self.batch_size)

        # Compile the model (using a simple optimizer and loss)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Train the model
        self.model.fit(dataset, epochs=self.epochs, verbose=1)
        return self

    def predict(self, X):
        if self.model is None:
            print("Loading best saved validated model...")
            self.model = tf.keras.models.load_model('best_validated_model')
        tokenized = self._tokenize(X)
        dataset = tf.data.Dataset.from_tensor_slices(dict(tokenized)).batch(self.batch_size)
        
        output = self.model.predict(dataset)
    
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output.logits
        
        predictions = np.argmax(logits, axis=1)
        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
    
    def scores(self, X, y):
        preds = self.predict(X)
        testScores = [accuracy_score(y, preds),
                      precision_score(y, preds),
                      recall_score(y, preds),
                      f1_score(y, preds)]
        return testScores

if __name__ == "__main__":
    ### Initialize and Clean Dataset
    # Load dataset & Handle categorical variables
    df = pd.read_csv('Phishing_Email.csv', header=0, sep=',')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.columns = ['BodyText', 'Type_isPhishing']
    df['Type_isPhishing'] = df['Type_isPhishing'].map({'Phishing Email': 1, 'Safe Email': 0})

    ### Split into "CrossVal Train / Final Test" sets
    # Split the data (80% for CrossVal and 20% for Test)
    trainCVset, testFset = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['Type_isPhishing']) 
    
    # Split Features and Targets
    X_trainCVset = trainCVset['BodyText']
    y_trainCVset = trainCVset['Type_isPhishing']
    X_testFset = testFset['BodyText']
    y_testFset = testFset['Type_isPhishing']
    
    ### Cross-validation using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_score = -np.inf
    cv_scores = []
    fold = 1
    
    for train_index, val_index in skf.split(X_trainCVset, y_trainCVset):
        print(f"\nStarting fold {fold}...")
        X_tr, X_val = X_trainCVset.iloc[train_index], X_trainCVset.iloc[val_index]
        y_tr, y_val = y_trainCVset.iloc[train_index], y_trainCVset.iloc[val_index]

        # Initialize the classifier (each fold gets a fresh model)
        clf = TransformerClassifier()
        clf.fit(X_tr, y_tr)
        
        # Evaluate on the validation fold
        score = clf.score(X_val, y_val)
        print(f"Fold {fold} validation accuracy: {score:.4f}")
        cv_scores.append(score)

        # Save best validation model
        if score > best_score:
            best_score = score
            clf.model.save('best_validated_model')
            print("New best model found!")

        fold += 1

    print("\nCross-validation scores:", cv_scores)
    print("Mean CV accuracy: {:.4f}".format(np.mean(cv_scores)))
    
    ### Evaluate on the final test set
    test_model = TransformerClassifier(batch_size=1)
    test_result = test_model.scores(X_testFset, y_testFset)
    print("\nFinal test set accuracy: {:.4f}".format(test_result[0]))
    print("\nFinal test set precision: {:.4f}".format(test_result[1]))
    print("\nFinal test set recall: {:.4f}".format(test_result[2]))
    print("\nFinal test set f1-score: {:.4f}".format(test_result[3]))
