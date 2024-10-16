import os.path
import chromadb
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex,
    PromptTemplate,
    set_global_tokenizer,
    Settings,
    StorageContext,
)
from transformers import AutoTokenizer
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

def initialize_conversation_bot():
    llm = LlamaCPP(
        model_url=None,
        model_path=r"K:\project-CMK\llama-2-13b-chat.Q8_0.gguf",
        temperature=0.1,
        max_new_tokens=1024,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 100},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    global documents, vector_index, PERSIST_DIR, storage_context
    PERSIST_DIR = "./Storage"

    # load the documents and create the index
    documents = SimpleDirectoryReader("Data").load_data(show_progress=True)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("etta-dt")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    vector_index.storage_context.persist(persist_dir=PERSIST_DIR)

    template = (
        "Imagine you are a personal healthcare assistant with personal name Etta, with access to medical reports, "
        "Your goal is to suggest tips or treatments using the given reports or just engage in a conservation if needed(say hi if the question is hi)\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following query, "
        "User: {query_str}\n\n"
        "When you switch to next line, use '\\n' at the end of sentence:\n\n"
        "Ensure your response suits the question,don't just say hello for every message and don't just look answers within context and reply general comments like a chatbot"
    )
    qa_template = PromptTemplate(template)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    query_engine = vector_index.as_query_engine(
        chat_mode="context",
        llm=llm,
        memory=memory,
        system_prompt=qa_template,
    )

    return query_engine

def update_index():
    documents = SimpleDirectoryReader("Data").load_data()
    vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    vector_index.storage_context.persist(persist_dir=PERSIST_DIR)

def chat_with_bot(query_engine, question):
    response_stream = query_engine.stream_chat(question)
    return response_stream

query_engine = initialize_conversation_bot()
response = chat_with_bot(query_engine, question:=input("You: "))
for token in response.response_gen:
    print(token, end="")
del response, query_engine
