# Backend module for chatbot functionality
# Consider splitting this into multiple modules? There are several options for each part of the chatbot (vectorstore, llm, language/dictionary, )
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint, AzureMLEndpointApiType
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_community.document_loaders import JSONLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from huggingface_hub import login

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
        model=model_name,
        temperature=0.7,
        api_key=api_key # this should be an environmental variable later
    )
    return chat_model

def initialize_chatbot_azureml(endpoint_url: str, api_key: str, api_type: str = "dedicated"):
    # Map string to AzureMLEndpointApiType
    api_type_enum = {
        "dedicated": AzureMLEndpointApiType.dedicated, # dedicated for online endpoints
        "serverless": AzureMLEndpointApiType.serverless, # serverless for pay-as-you-go model as a service
    }.get(api_type.lower(), AzureMLEndpointApiType.dedicated)

    # Initialize AzureMLOnlineEndpoint model
    chat_model = AzureMLChatOnlineEndpoint(
        endpoint_url=endpoint_url,
        endpoint_api_key=api_key,
        endpoint_api_type=api_type_enum
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

# Local faiss vector store
def initialize_faiss(documents, embeddings: Embeddings):
    # Initialize FAISS vector store from documents and embeddings
    faiss_store = FAISS.from_documents(documents, embeddings)
    return faiss_store

def initialize_cosmos_db(documents,             
    embedding: Embeddings, 
    cosmos_client: str,
    vector_embedding_policy: dict,
    indexing_policy: dict,
    cosmos_container_properties: dict,
    cosmos_database_properties: dict,
    full_text_policy: dict = None,
    database_name: str = "vectorSearchDB",
    container_name: str = "vectorSearchContainer",):
    db = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=cosmos_client,
        embedding=embedding,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties=cosmos_database_properties,
        full_text_policy=full_text_policy,
        database_name=database_name,
        container_name=container_name,
    )
    db.add_documents(documents)
    return db

def load_existing_cosmos_db(             
    embedding: Embeddings, 
    cosmos_client: str,
    vector_embedding_policy: dict,
    indexing_policy: dict,
    cosmos_container_properties: dict,
    cosmos_database_properties: dict,
    full_text_policy: dict = None,
    database_name: str = "vectorSearchDB",
    container_name: str = "vectorSearchContainer",):
    db = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=cosmos_client,
        embedding=embedding,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties=cosmos_database_properties,
        full_text_policy=full_text_policy,
        database_name=database_name,
        container_name=container_name,
    )
    return db




def initialize_conversation_partner(
    llm: BaseChatModel,
    db: VectorStore,
):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"search_type": "similarity", "k": 5})
    
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = retriever.invoke(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. Use the following context in your response:" # Change this for the conversation partner
            f"\n\n{docs_content}"
        )

        return system_message
    
    
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[prompt_with_context]
    )
    return agent

if __name__ == "__main__":
    docs = load_dictionary_jp("n5.json")
    print(f"Loaded {len(docs)} dictionary entries.")
    for doc in docs[:5]:
        print(doc.page_content)
    
    