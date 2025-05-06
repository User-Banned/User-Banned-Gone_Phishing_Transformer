# Overview

This repository contains two key components for the Gone Phishing Extension demo, which showcases a feature designed to enhance phishing detection capabilities. Please note that the models generated during the training process are not included in this repository due to their large size.

# Folder Structure
## The project is organized into two main folders:

### 1. transformersPY Folder

This folder contains the Python code, which includes:

**main.py :** The core functionality for training the transformer model is implemented here.

**predict_text.py :** A separate file to test email text in an enclosed enviorment.

**getTokenizer.py :** Used to get the tokenizer to be used in the conversion file.

**req.txt :** Holds the required package versions that worked for the conversion.

### 2. transformersJS Folder

This folder contains the JavaScript code, which includes:

**Trasformer.js :** A JavaScript test example that tests the model before integration into the browser extension. This allows the verification the model's functionality and performance in a controlled environment. 

## Note

Due to the size of the generated models, you will need to either generate a new batch or include them separately. Ensure that you have the necessary dependencies installed for both Python and JavaScript components. And there may need to be modifications to some code when converting the model from PY to JS.
