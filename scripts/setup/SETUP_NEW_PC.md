# Setting up fair-teacher-fair-student on a new PC (RTX 5090 / WSL2)

Do the **Windows host** steps first, then the **inside-WSL** steps. The GPU driver
lives on Windows; you never install a GPU driver or CUDA toolkit inside WSL.

## A. Windows host (do these first)

1. **Windows 11, or Windows 10 version 21H2+.**
2. **Install the NVIDIA CUDA-enabled driver for WSL.** For an RTX 5090 (Blackwell)
   this must be a **570+ / Blackwell driver**. Get it from
   https://www.nvidia.com/download/index.aspx (a standard up-to-date GeForce driver
   includes WSL CUDA support — you do NOT separately install CUDA on Windows).
3. **Update WSL** (PowerShell):
   ```powershell
   wsl --update
   wsl --shutdown
   ```
   Kernel must be ≥ 5.10.43.3. Check with: `wsl cat /proc/version`.
4. Install a glibc distro (Ubuntu/Debian) if you don't have one: `wsl --install -d Ubuntu`.

> Ref: Microsoft, "Enable NVIDIA CUDA on WSL 2"; NVIDIA "CUDA on WSL User Guide".

## B. Inside WSL (after the Windows steps)

1. Confirm the GPU is visible:
   ```bash
   nvidia-smi          # should list "NVIDIA GeForce RTX 5090"
   ```
   If this fails, the Windows driver step (A2) isn't done — fix that first.

2. From the project root, run the setup script:
   ```bash
   cd fair-teacher-fair-student                      # the extracted project folder
   ./scripts/setup/setup_5090.sh    # cd's to repo root itself; run from anywhere
   ```
   This makes a venv, installs **Blackwell-capable torch (cu128 wheels)**, installs
   the rest from `scripts/setup/requirements-5090.txt`, and verifies a real GPU kernel launches.

   > Do NOT `pip install -r requirements.txt` (the original) — it pins
   > torch 2.4.1+cu121, which has no kernels for the 5090 and will fail.

3. Run the experiment:
   ```bash
   source venv/bin/activate
   ./scripts/pipelines/run_multiseed_robustness_experiment.sh
   ```

## Notes

- **First run downloads BERT / BERT-tiny** from HuggingFace — the machine needs
  internet on the first run. (The 20 GB HF cache from the old machine was not copied.)
- **Resumable**: re-running the experiment skips any seed/method whose checkpoint
  and results already exist.
- If `setup_5090.sh`'s verify step prints `RTX 5090`, `sm_120`, and `kernel launch: OK`,
  you're clear. If it fails there, it's the Windows-side driver (too old for Blackwell),
  not the Python environment.
- If PyTorch ever can't find cu128 wheels, check the current Blackwell install command
  at https://pytorch.org/get-started/locally/ (cu128 is the Blackwell target as of now).
