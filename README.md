# END to END AIMediBot using LLAMA2
AIMediBot is a sophisticated AI-driven medical chatbot designed to assist patients by providing preliminary medical advice and answering health-related queries. This end-to-end solution leverages advanced AI models and a robust tech stack, including Python, LangChain, Flask, Meta's Llama2, and Pinecone, to ensure seamless and efficient operation.
# How to run?

### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone
