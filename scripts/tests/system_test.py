import os
import torch
import deepspeed
from accelerate import Accelerator
from accelerate.state import PartialState
import json


# Environment variables for distributed training
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "8084"  # Custom port from your setup
os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
os.environ["RANK"] = "0"
# Set threading environment variables
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

def check_torch_and_cuda():
    """Check PyTorch and CUDA compatibility."""
    print("========== PyTorch and CUDA Test ==========")
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version (from PyTorch): {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"    - Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"Error testing PyTorch and CUDA: {e}")


def check_deepspeed_config():
    """Verify DeepSpeed configuration file exists and is valid."""
    print("\n========== DeepSpeed Configuration Test ==========")
    try:
        ds_config_path = './ds_config_zero2.json'
        if not os.path.exists(ds_config_path):
            raise FileNotFoundError(f"DeepSpeed config file not found at {ds_config_path}")

        with open(ds_config_path, 'r') as f:
            config = json.load(f)
            print(f"DeepSpeed config loaded successfully: {config}")

        # Check essential settings
        if "fp16" not in config or not config["fp16"]["enabled"]:
            print("Warning: FP16 is not enabled in DeepSpeed config.")
        else:
            print("FP16 is enabled in DeepSpeed config ✅")
    except Exception as e:
        print(f"Error checking DeepSpeed configuration: {e}")


def check_accelerate():
    print("\n========== Accelerate Test ==========")
    try:
        # Initialize Accelerator with debugging
        accelerator = Accelerator()
        print("Accelerate initialized successfully ✅")
        print(f"Using device: {accelerator.device}")

        # Perform a simple tensor computation
        tensor = torch.tensor([1.0, 2.0, 3.0], device=accelerator.device)
        result = tensor + 1
        print(f"Tensor computation result: {result}")
    except Exception as e:
        print(f"Error testing Accelerate: {e}")


def check_environment_variables():
    """Verify necessary environment variables for distributed training."""
    print("\n========== Environment Variable Test ==========")
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            print(f"Error: Environment variable {var} is not set!")
        else:
            print(f"{var}={value}")


def check_model_params():
    """Verify model parameters align with the expected configuration."""
    print("\n========== Model Parameter Test ==========")
    try:
        model_name = "TimeLLM"
        train_epochs = 10
        learning_rate = 0.01
        llama_layers = 32
        batch_size = 24
        d_model = 32
        d_ff = 32

        print(f"Model Name: {model_name}")
        print(f"Train Epochs: {train_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"LLM Layers: {llama_layers}")
        print(f"Batch Size: {batch_size}")
        print(f"d_model: {d_model}, d_ff: {d_ff}")

        # Add any additional checks for parameters if needed
        if d_model < 16 or d_ff < 16:
            print("Warning: d_model or d_ff is smaller than recommended values.")
    except Exception as e:
        print(f"Error checking model parameters: {e}")


def check_disk_and_memory():
    """Ensure sufficient disk space and memory are available."""
    print("\n========== Disk and Memory Test ==========")
    try:
        # Check available GPU memory
        mem_info = torch.cuda.get_device_properties(0)
        print(f"Available GPU memory: {mem_info.total_memory / 1e9:.2f} GB")

        # Check disk space
        st = os.statvfs('.')
        free_space = (st.f_bavail * st.f_frsize) / 1e9
        print(f"Available disk space: {free_space:.2f} GB")
    except Exception as e:
        print(f"Error checking disk or memory: {e}")


def check_dependencies():
    """Verify essential dependencies are installed and correctly versioned."""
    print("\n========== Dependency Test ==========")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers library is not installed!")

    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("Datasets library is not installed!")


def main():
    print("Starting system tests...\n")
    check_torch_and_cuda()
    check_deepspeed_config()
    # check_accelerate()
    check_environment_variables()
    # check_model_params()
    # check_disk_and_memory()
    # check_dependencies()


if __name__ == "__main__":
    main()
