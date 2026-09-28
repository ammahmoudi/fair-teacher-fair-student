import logging
import os
from transformers import GPT2Model, LlamaModel, BertModel, GPT2Config, LlamaConfig, BertConfig
from transformers import GPT2Tokenizer, LlamaTokenizer, BertTokenizer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_model_availability(model_name, model_type, config=None, local_only=True):
    """
    Check if the model is available locally, and log useful debugging info.

    model_name: str - The model name or local path to the model.
    model_type: str - The model type ('GPT2', 'LLAMA', or 'BERT').
    config: dict - Configuration settings (optional).
    local_only: bool - Whether to check only locally available models.
    """
    try:
        if model_type == 'GPT2':
            logger.debug(f"Attempting to load GPT2 model: {model_name}")
            model = GPT2Model.from_pretrained(model_name, local_files_only=local_only)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_only)
        elif model_type == 'LLAMA':
            logger.debug(f"Attempting to load LLAMA model: {model_name}")
            model = LlamaModel.from_pretrained(model_name, local_files_only=local_only)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, local_files_only=local_only)
        elif model_type == 'BERT':
            logger.debug(f"Attempting to load BERT model: {model_name}")
            model = BertModel.from_pretrained(model_name, local_files_only=local_only)
            tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=local_only)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.debug(f"Model {model_name} loaded successfully.")
        return model, tokenizer
    except EnvironmentError as e:
        logger.error(f"Failed to load model {model_name}. Error: {str(e)}")
        if not local_only:
            logger.info(f"Trying to load model remotely if local files are missing.")
            return check_model_availability(model_name, model_type, config=config, local_only=False)
        else:
            logger.debug(f"Model {model_name} is not found locally. Check the path or model name.")
            return None, None
    except Exception as e:
        logger.error(f"Unexpected error while loading {model_name}: {str(e)}")
        return None, None


def check_all_models(configs):
    """
    Check availability of LLAMA, GPT2, and BERT models based on the configuration.
    
    configs: dict - Configuration containing model settings (model names, types).
    """
    model_types = ["LLAMA", "GPT2", "BERT"]
    for model_type in model_types:
        model_name = configs.get(f"{model_type}_model_name")
        if model_name:
            logger.info(f"Checking availability for {model_type} model: {model_name}")
            model, tokenizer = check_model_availability(model_name, model_type)
            if model:
                logger.info(f"{model_type} model {model_name} is available.")
            else:
                logger.warning(f"{model_type} model {model_name} is not available.")
        else:
            logger.warning(f"No {model_type} model name found in configuration.")


def main():
    # Configuration settings for the models
    configs = {
        "LLAMA_model_name": "huggyllama/llama-7b",  # Replace with the LLAMA model name
        "GPT2_model_name": "openai-community/gpt2",  # Replace with the GPT2 model name
        "BERT_model_name": "google-bert/bert-base-uncased",  # Replace with the BERT model name
    }
    
    # Check availability for LLAMA, GPT2, and BERT models
    check_all_models(configs)


if __name__ == "__main__":
    main()
