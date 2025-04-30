# MPEP LLaMA Fine-Tuning

This repository contains scripts and data for fine-tuning the LLaMA 3.1 8B model on Multiple Choice Questions covering the **Manual of Patent Examining Procedure (MPEP)**.

## 🚀 Project Goals
- Improve LLaMA's ability to answer patent law-related multiple choice questions.
- Train on structured **question-answer (QA) pairs**.
- Utilize **LoRA/QLoRA** to efficiently fine-tune on a GPU.

## 📂 Repository Structure
```
📂 mpep-llama-finetune
 ├── notebooks/                # Jupyter notebooks for preprocessing and analysis
 ├── scripts/                  # Python scripts for fine-tuning and dataset processing
 ├── datasets/                 # Training data (ignored in GitHub)
 ├── models/                   # Checkpoints (ignored in GitHub)
 ├── config.json               # Model training configuration
 ├── .gitignore                # Files to exclude from Git tracking
 ├── README.md                 # Project documentation
```

## 🛠️ Setup
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Run Preprocessing
```sh
python scripts/process_mpep_json.py
```

### 3️⃣ Start Fine-Tuning
```sh
python scripts/finetune_llama3.py
```

## 📊 Expected Results
- The fine-tuned model should better understand **patent law terminology**.
- It should answer **MPEP-related questions** with higher accuracy.

## 📌 TODO
- ✅ Add dataset processing scripts
- ✅ Implement LoRA fine-tuning
- 🚀 Optimize model inference for fast response times

## 🤝 Contributing
Feel free to submit issues or pull requests if you have improvements!

## 📜 License
This project is open-source under the **MIT License**.
