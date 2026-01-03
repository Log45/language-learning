# Backend module for chatbot functionality
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint, AzureMLEndpointApiType
from langchain_community.document_loaders import JSONLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from huggingface_hub import login
import torch

# local model
def initialize_chatbot_hf(model_name: str, hf_token: str):
    # Log in to Hugging Face
    login(hf_token)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create HuggingFace pipeline
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        model_kwargs={"temperature": 0.7}
    )

    hf = HuggingFacePipeline(pipeline=pipe)
    # Initialize ChatHuggingFace with the pipeline
    chat_model = ChatHuggingFace(llm=hf)

    return chat_model

# Cloud models
def initialize_chatbot_openai(api_key: str, model_name: str = "gpt-3.5-turbo"):
    # Initialize ChatOpenAI model
    chat_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        openai_api_key=api_key
    )
    return chat_model

def initialize_chatbot_azureml(endpoint_url: str, api_key: str, api_type: str = "dedicated"):
    # Map string to AzureMLEndpointApiType
    api_type_enum = {
        "dedicated": AzureMLEndpointApiType.dedicated, # dedicated for online endpoints
        "serverless": AzureMLEndpointApiType.serverless, # serverless for pay-as-you-go model as a service
    }.get(api_type.lower(), AzureMLEndpointApiType.dedicated)

    # Initialize AzureMLOnlineEndpoint model
    chat_model = AzureMLOnlineEndpoint(
        endpoint_url=endpoint_url,
        api_key=api_key,
        api_type=api_type_enum
    )
    return chat_model

def load_dictionary_jp(file_path: str):
    # Load dictionary data from JSON file using langchain's JSONLoader.
    jq_schema = ".[]"
    # for now just take the kanji since it should only be used in reference by japanese llm
    content_key = '(.kanji)'
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,
        content_key=content_key,
        is_content_key_jq_parsable=True,
    )
    documents = loader.load()
    return documents

def load_dictionaries(file_paths: list[str], language: str = "jp"):
    # Load multiple dictionary files and combine their entries. (if user wants multiple levels)
    all_documents = []
    for file_path in file_paths:
        if language == "jp":
            docs = load_dictionary_jp(file_path)
            all_documents.extend(docs)
        else:
            # Placeholder for other languages
            pass
    return all_documents

if __name__ == "__main__":
    docs = load_dictionary_jp("n5.json")
    print(f"Loaded {len(docs)} dictionary entries.")
    for doc in docs[:5]:
        print(doc.page_content)
    
    