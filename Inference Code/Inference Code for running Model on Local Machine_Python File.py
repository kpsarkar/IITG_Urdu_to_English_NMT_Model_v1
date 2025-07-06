# Inference Code using a offline Local Machine

# Install necessary packages if not already installed
# pip install transformers sentencepiece torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the saved model and tokenizer
model_path = "./save_model-finetuned-ur-to-en"  # Adjust this path to your local folder
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define a function for translation
def translate_text(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation using the model
    translated_outputs = model.generate(**inputs)
    # Decode the translated text
    translated_text = tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
    return translated_text

# Human Evaluation
if __name__ == "__main__":
    input_text = "میں گاڑی چلانا چاہتا ہوں۔"
    translated = translate_text(input_text)
    print("Input (Urdu):", input_text)
    print("Translated (English):", translated)
