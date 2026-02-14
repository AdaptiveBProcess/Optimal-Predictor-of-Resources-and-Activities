# 🐍 Python 3.11.14 Environment Setup (Conda)

> Uses **Conda** to fully isolate Python **3.11.14**
> No global Python install required.

---

## 0️⃣ Make sure Conda is installed

Check:

```bash
conda --version
```

If you don’t have it:

* Install **Miniconda** (recommended) or Anaconda
* Restart your terminal after installation

---

## 1️⃣ Create a Conda environment (Python 3.11.14)

```bash
conda create -n opra_env python=3.11.14
```

Activate it:

```bash
conda activate opra_env
```

✅ Your prompt should now look like:

```text
(opra_env)
```

Verify:

```bash
python --version
# Python 3.11.14
```

---

## 2️⃣ (Optional but recommended) Set channels

For better package compatibility on Windows:

```bash
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

---

## 3️⃣ Install dependencies

### 🔥 PyTorch (CUDA 12.6)

> Only if you **have an NVIDIA GPU + CUDA 12.6**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

### 📦 Project requirements

Install `requirements.txt`, **pip-based**:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy matplotlib
```

---

## 4️⃣ Verify everything

```bash
python --version
pip list
```

---

## 5️⃣ Deactivate when done

```bash
conda deactivate
```