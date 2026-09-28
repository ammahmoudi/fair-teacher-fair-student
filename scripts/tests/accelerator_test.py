import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CURL_CA_BUNDLE"] = ""

# Fix seed for reproducibility
set_seed(42)

def test_accelerate():
    """Test Accelerate with basic model and dataset."""
    print("========== Accelerate Test ==========")
    try:
        # Define DeepSpeed plugin
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')

        # Initialize Accelerator
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
        print(f"Accelerator initialized successfully on device: {accelerator.device}")

        # Dummy dataset
        X = torch.rand(100, 10)  # 100 samples, 10 features
        y = torch.randint(0, 2, (100, 1))  # Binary labels
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Dummy model
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        model = model.to(torch.bfloat16)  # Match precision

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Prepare components with Accelerator
        dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
        print("Data loader, model, and optimizer prepared successfully.")

        # Training loop
        model.train()
        for epoch in range(2):  # Run for 2 epochs
            epoch_loss = 0
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(accelerator.device, dtype=torch.bfloat16)  # Ensure correct dtype
                targets = targets.to(accelerator.device, dtype=torch.bfloat16)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accelerator.backward(loss)  # Use accelerator for backward pass
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

        print("Basic training loop completed successfully ✅")
    except Exception as e:
        print(f"Error in Accelerate test: {e}")


if __name__ == "__main__":
    test_accelerate()
