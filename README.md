# Private AI RAG Graph with Gemma, Ollama and LangGraph

Developing a **Private AI** application running **offline** with **Gemma 1B** Edge AI model using **Ollama**, **LangGraph**, and **RAG**.

# ![KrishiGPT Chatbot](./images/multiagentworking.png "KrishiGPT Chatbot")
# ![Ollama Pulling Models](./images/1.png "Pulling down both model")
# ![RAG based response](./images/2.png "RAG based response")
# ![RAG based continue chat](./images/3.png "RAG based continue chat")

---

## **Prerequisites**

- Python 3.11 or higher
- pip (Python package installer)
- (Optional) Google Cloud Platform (GCP) account (for Gemini API access)
- (Optional) LangSmith account (for tracing and debugging)

---

## **Installation**

1. **Fork and Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_application_directory>
    ```

2. **Create `.env` File:**

    Copy `.env_copy` to `.env`:

    ```
    cp env_copy .env
    ```

    Fill `.env` file:

    ```
    MODEL=gemma:1b
    BASE_URL=http://ollama_server:11434
    LANGCHAIN_API_KEY=<your_langsmith_api_key>  # Optional for tracing
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_PROJECT="Private_AI_Project"
    ```

---

## **Obtaining API Keys**

### 1. Google Gemini API Key (Optional)

- [Google AI Studio](https://makersuite.google.com/)
- [Google Cloud Console](https://cloud.google.com/)

> Enable Gemini API, create credentials, add to `.env` as `GOOGLE_API_KEY`.

### 2. LangSmith API Key (Optional)

- [LangSmith](https://smith.langchain.com/)

> Get API Key, add to `.env` as `LANGCHAIN_API_KEY`.

---

## **Running the Application**

1. **Using Docker Compose:**

    ```bash
    cd <your_application_directory>
    docker-compose up --build
    ```

2. **Using LangGraph CLI (Optional):**

    ```bash
    langgraph run --config langgraph.json
    # or
    langgraph dev
    ```

---

## **Development Notes**

- Enable LangSmith tracing by setting `LANGCHAIN_TRACING_V2="true"`.
- Check your runs visually in LangSmith UI.
- Add `.env` to your `.gitignore` to protect sensitive data.

> Helpful Links:
> - [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
> - [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
> - [LangSmith Documentation](https://docs.smith.langchain.com/)

---

## **Docker Compose Setup**

- **Run services**:

    ```bash
    docker-compose up --build
    ```

- **Stop services**:

    ```bash
    ctrl + c
    docker-compose down --remove-orphans
    ```

- **Services Launched**:
    - 🚀 `rag_service` on `http://localhost:5000`
    - 🧠 `ollama_server` on `http://localhost:11434`

- **System Requirements**:
    - Minimum 16GB RAM (32GB recommended)
    - i7 or similar CPU (GPU optional)
    - Disk space ~5GB for LLM model

> 🛑 Regularly clean Docker images/containers to avoid freezing issues!

