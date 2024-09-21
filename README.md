# Text Generation - Conversational Post Generator with LangChain and Transformers

This project is a Flask-based API that leverages LangChain, Hugging Face Transformers, and FAISS to generate engaging social media posts and detailed proposals. It supports various types of content generation, including Twitter posts, LinkedIn posts, formal proposals, and question-answering tasks. The API utilizes pre-trained language models such as Mistral-7B for generating responses based on historical context and user input.

---

## **Features**

- **Content Generation:** Generate engaging and contextually relevant posts for Twitter, LinkedIn, and formal proposals.
- **Question Answering:** Retrieve and respond to user queries based on historical context.
- **Customizable Outputs:** Adjust temperature and diversity (`top_p`) parameters to influence the model’s output.
- **History-Based Retrieval:** Uses FAISS for vectorized historical context retrieval to provide accurate and context-based answers.
- **Language Models:** Utilizes Hugging Face models, such as `Mistral-7B`, with 4-bit quantization for efficient and scalable text generation.

---

## **Tech Stack**

- **Backend Framework:** Flask (Python)
- **Machine Learning Libraries:**
  - LangChain
  - Transformers (Hugging Face)
  - FAISS for efficient retrieval
- **Deep Learning Framework:** PyTorch (CUDA)
- **Deployment:** Docker

---

## **How to Run the Project**

### **1. Clone the repository:**

```bash
git clone https://github.com/VRAJ-07/Text-Generation-Using-LLM.git
cd Text-Generation-Using-LLM
```

### **2. Set up the environment:**

Ensure you have Docker installed, then build and run the Docker container.

```bash
docker build -t text-generation .
docker run -p 5010:5010 text-generation
```

### **3. API Endpoints:**

**POST /conversation**
- **Description:** Generates the response for social media posts or question-answering based on user input.
- **Request Payload:**

```json
{
  "prompt": "Generate a post about AI development in healthcare",
  "model_name": "mistral-7B",
  "contentfor": "linkedin",
  "temp": 0.7,
  "top_p": 0.85
}
```

- **Response:**

```json
{
  "response": "Exciting advancements are happening in AI healthcare... #AI #Healthcare"
}
```

---

## **Model Configuration**

The project uses the Mistral-7B-Instruct-v0.2 model. You can adjust the model configurations in the `run_model` function in `app.py` for different quantization settings or model paths.

---

## **Docker Setup**

The Dockerfile sets up the environment with PyTorch, LangChain, FAISS, and other dependencies. To modify model paths or configurations, you can update the DockerFile or `app.py`.

---

## **Dependencies**

- `langchain`
- `langchain_community`
- `transformers`
- `sentence-transformers`
- `flask`
- `diffusers`

Install them via:

```bash
pip install -r requirements.txt
```

---

This README should work well for your project and provide the necessary information for others to use or contribute to it! Let me know if you need any adjustments.
