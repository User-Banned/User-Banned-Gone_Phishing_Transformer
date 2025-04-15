import { env, AutoModel, AutoTokenizer } from '@huggingface/transformers';

env.localModelPath = './models';    // Specify a custom location for models (defaults to '/models/').
env.allowRemoteModels = false;      // Disable the loading of remote models from the Hugging Face Hub:
env.backends.onnx.wasm.wasmPaths;   // Set location of .wasm files. Defaults to use a CDN.
const tokenizer = await AutoTokenizer.from_pretrained('distilbert-base-uncased');   // Load Local Tokenizer
const model = await AutoModel.from_pretrained('distilbert-base-uncased');           // Load Local Model

async function predictText(text) {
    // Truncate the text if it exceeds the character limit
    const MAX_CHAR_LIMIT = 512;
    if (text.length > MAX_CHAR_LIMIT) {
        text = text.substring(0, MAX_CHAR_LIMIT);
    }

    let inputs = await tokenizer(text);
    
    // Truncate the tokens if it exceeds the token limit
    const MAX_TOKEN_LIMIT = 512;
    if (inputs.input_ids.length > MAX_TOKEN_LIMIT) {
        inputs.input_ids = inputs.input_ids.slice(0, MAX_TOKEN_LIMIT);
        inputs.attention_mask = inputs.attention_mask.slice(0, MAX_TOKEN_LIMIT);
    }

    let { logits } = await model(inputs);

    const prediction = logits[0].indexOf(Math.max(...logits[0]));

    return prediction;  // Reminder outputs { 0 : Safe , 1 : Phishing }
}

// Example Use (copy this block for own use)
async function main() {
    const text = `Input Email Text Here.`; // Example input text
    const prediction = await predictText(text); // Get the prediction
    console.log("Predicted class:", prediction); // Output the predicted class
}

//ain(); // Test Use