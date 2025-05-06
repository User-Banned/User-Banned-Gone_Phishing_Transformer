# Overview

This project provides a framework for training and testing a model for text prediction, specifically tailored for email text editing. The code is designed to be user-friendly, allowing for easy installation and execution.

## Installation

To get started, ensure you have Python and pip installed on your machine. Then, install the required dependencies by running the following command:

    pip install -r req.txt

This command will install all the necessary packages listed in the req.txt file.

## Usage
### Training and Testing

To run the training and testing portion of the code, execute the following command:

    python main.py

This will initiate the training process and evaluate the model based on the provided dataset.

### Predicting Email Text

To test the model with a specific email text, follow these steps:

1. Open the predict_text.py file.
2. Locate the section within the triple quotes that says "REPLACE TEXT HERE".
3. Replace this placeholder text with the body of the email you wish to analyze.

After making the necessary changes, run the prediction script with the following command:

    python predict_text.py

This will output the model's predictions based on the provided email text.

## Converting to JavaScript

Converting the model to JavaScript can be challenging due to the various methods available. The approach used in this project follows the @huggingface/transformersjs conversion method. For detailed guidance on this method, refer to the official documentation here: https://huggingface.co/docs/transformers.js/custom_usage.

Please note that the conversion process may involve several trials and errors.
