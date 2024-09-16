from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompt_values import ChatPromptValue
from llama_parse import LlamaParse


from langchain_groq import ChatGroq

import nest_asyncio; nest_asyncio.apply()
import time

import pickle
import os
from dotenv import load_dotenv

load_dotenv()


def parse_pdfs(documents_dir):

    # set up parser
    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        language="de",
    )

    file_paths = [os.path.join(documents_dir, filename)
                  for filename in os.listdir(documents_dir)
                  if filename.endswith('.pdf') and os.path.isfile(os.path.join(documents_dir, filename))]

    for file_path in file_paths:
        # sync batch
        document = parser.load_data(file_path)
        # Save the markdown content to a file

        file_path = file_path.split(".pdf")[0]
        markdown_file_path = file_path + '.md'  # Define the path where you want to save the Markdown file
        with open(markdown_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(document[0].text)  # Ensure the content is converted to string if it's not already

        print(f"PDF is parsed and the markdown content is saved to {markdown_file_path}")


def get_all_material_retrieval(embeddings_model):
    global_storage_dir = "storage/data/all_material_index/"
    # Load or generate the index for all lectures

    try:
        # Load the object back from the file
        with open(global_storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever = pickle.load(file)

        faiss_vectorstore = FAISS.load_local(global_storage_dir + "vector_store", embeddings_model,
                                             allow_dangerous_deserialization=True)

    except:
        print("Combining indices and saving it")

        storage_dir = "storage/data/lecture_material_index/"

        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever = pickle.load(file)

        faiss_vectorstore = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                             allow_dangerous_deserialization=True)

        storage_dir = "storage/data/organisational_information_index/"
        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever_organisational = pickle.load(file)

        faiss_vectorstore_organisational = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                                            allow_dangerous_deserialization=True)

        faiss_vectorstore.merge_from(faiss_vectorstore_organisational)

        bm25_retriever_docs = bm25_retriever.docs

        bm25_retriever_docs.extend(bm25_retriever_organisational.docs)

        storage_dir = "storage/data/seminar_text_index/"
        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever_seminar = pickle.load(file)

        faiss_vectorstore_seminar = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                                     allow_dangerous_deserialization=True)

        faiss_vectorstore.merge_from(faiss_vectorstore_seminar)

        bm25_retriever_docs = bm25_retriever.docs

        bm25_retriever_docs.extend(bm25_retriever_seminar.docs)

        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_documents(
            bm25_retriever_docs
        )

        # Ensure the storage directory exists
        if not os.path.exists(global_storage_dir):
            os.makedirs(global_storage_dir, exist_ok=True)  # Creates the directory if it doesn't exist

        with open(global_storage_dir + 'bm25_retriever.pkl', 'wb') as file:
            pickle.dump(bm25_retriever, file)

        faiss_vectorstore.save_local(global_storage_dir + "vector_store")

    bm25_retriever.k = 50
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 50})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = CohereRerank(top_n=5, model="rerank-english-v3.0")
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compressor_retriever


def get_lecture_material_retrieval(embeddings_model):
    documents_dir = "data/lecture_material/"
    storage_dir = "storage/data/lecture_material_index/"
    # Load or generate the index for all lectures

    try:
        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever = pickle.load(file)

        faiss_vectorstore = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                             allow_dangerous_deserialization=True)


    except:
        print("Generating index and save it in storage")

        file_paths = [os.path.join(documents_dir, filename)
                      for filename in os.listdir(documents_dir)
                      if filename.endswith('.md') and os.path.isfile(os.path.join(documents_dir, filename))]

        documents = []

        for file_path in file_paths:
            print("Splitting and embedding the file: " + file_path)
            loader = UnstructuredMarkdownLoader(file_path, mode="single", strategy="fast")
            data = loader.load()

            chunk_size = 1024
            chunk_overlap = 20
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",
                                                                                 chunk_size=chunk_size,
                                                                                 chunk_overlap=chunk_overlap,
                                                                                 )

            # Split
            splits = text_splitter.create_documents([data[0].page_content])

            for split in splits:
                split.metadata["file_name"] = file_path.split("/")[-1]
                split.metadata["lecture_number"] = int(file_path.split("/")[-1].split(" ")[-1].split(".")[0])
                split.page_content = "Dokumentname: " + split.metadata["file_name"] + "\n\nVorlesungsnummer: " + str(
                    split.metadata["lecture_number"]) + "\n\nDokumentinhalt: \n\n" + split.page_content

            documents.extend(splits)

        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        # Ensure the storage directory exists
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)  # Creates the directory if it doesn't exist

        with open(storage_dir + 'bm25_retriever.pkl', 'wb') as file:
            pickle.dump(bm25_retriever, file)

        faiss_vectorstore = FAISS.from_documents(
            documents, embeddings_model
        )
        faiss_vectorstore.save_local(storage_dir + "vector_store")

    bm25_retriever.k = 50
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 50})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = CohereRerank(top_n=5, model="rerank-multilingual-v3.0")
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compressor_retriever


def get_seminar_material_retrieval(embeddings_model):
    documents_dir = "data/seminar_text/"
    storage_dir = "storage/data/seminar_text_index/"
    # Load or generate the index for all lectures

    try:
        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever = pickle.load(file)

        faiss_vectorstore = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                             allow_dangerous_deserialization=True)

    except:
        print("Generating index and save it in storage")
        file_paths = [os.path.join(documents_dir, filename)
                      for filename in os.listdir(documents_dir)
                      if filename.endswith('.md') and os.path.isfile(os.path.join(documents_dir, filename))]

        documents = []

        for file_path in file_paths:
            print("Splitting and embedding the file: " + file_path)
            loader = UnstructuredMarkdownLoader(file_path, mode="single", strategy="fast")
            data = loader.load()

            chunk_size = 1024
            chunk_overlap = 20
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",
                                                                                 chunk_size=chunk_size,
                                                                                 chunk_overlap=chunk_overlap,
                                                                                 )

            # Split
            splits = text_splitter.create_documents([data[0].page_content])

            for split in splits:
                split.metadata["file_name"] = file_path.split("/")[-1]
                if len(file_path.split("/")[-1].split(" ")) > 1:
                    split.metadata["seminar_number"] = int(file_path.split("/")[-1].split(" ")[1])
                else:
                    split.metadata["seminar_number"] = 0
                split.page_content = "Dokumentname: " + split.metadata["file_name"] + "\n\nSeminarnummer: " + \
                                     str(split.metadata[
                                             "seminar_number"]) + "\n\nDokumentinhalt: \n\n" + split.page_content

            documents.extend(splits)

        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        # Ensure the storage directory exists
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)  # Creates the directory if it doesn't exist

        with open(storage_dir + 'bm25_retriever.pkl', 'wb') as file:
            pickle.dump(bm25_retriever, file)

        faiss_vectorstore = FAISS.from_documents(
            documents, embeddings_model
        )
        faiss_vectorstore.save_local(storage_dir + "vector_store")

    bm25_retriever.k = 50
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 50})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = CohereRerank(top_n=5, model="rerank-multilingual-v3.0")
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compressor_retriever


def get_organisational_material_retrieval(embeddings_model):

    documents_dir = "data/organisational_information/"
    storage_dir = "storage/data/organisational_information_index/"
    # Load or generate the index for all lectures

    try:
        # Load the object back from the file
        with open(storage_dir + 'bm25_retriever.pkl', 'rb') as file:
            bm25_retriever = pickle.load(file)

        faiss_vectorstore = FAISS.load_local(storage_dir + "vector_store", embeddings_model,
                                             allow_dangerous_deserialization=True)

    except:
        print("Generating index and save it in storage")

        parse_pdfs(documents_dir)

        file_paths = [os.path.join(documents_dir, filename)
                      for filename in os.listdir(documents_dir)
                      if filename.endswith('.md') and os.path.isfile(os.path.join(documents_dir, filename))]

        documents = []

        for file_path in file_paths:
            print("Splitting and embedding the file: " + file_path)
            loader = UnstructuredMarkdownLoader(file_path, mode="single", strategy="fast")
            data = loader.load()

            chunk_size = 1024
            chunk_overlap = 20
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",
                                                                                 chunk_size=chunk_size,
                                                                                 chunk_overlap=chunk_overlap,
                                                                                 )

            # Split
            splits = text_splitter.create_documents([data[0].page_content])

            for split in splits:
                split.metadata["file_name"] = file_path.split("/")[-1]
                split.metadata["file_type"] = file_path.split("/")[-1].split(" ")[0]
                split.page_content = "Dokumentname: " + split.metadata["file_name"] + "\n\nDokumentart: " + \
                                     split.metadata["file_type"] + "\n\nDokumentinhalt: \n\n" + split.page_content

            documents.extend(splits)

        # initialize the bm25 retriever and faiss retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents
        )

        # Ensure the storage directory exists
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)  # Creates the directory if it doesn't exist

        with open(storage_dir + 'bm25_retriever.pkl', 'wb') as file:
            pickle.dump(bm25_retriever, file)

        faiss_vectorstore = FAISS.from_documents(
            documents, embeddings_model
        )
        faiss_vectorstore.save_local(storage_dir + "vector_store")

    bm25_retriever.k = 50
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 50})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    compressor = CohereRerank(top_n=5, model="rerank-multilingual-v3.0")
    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compressor_retriever


def get_retrieval(tool_name, embeddings_model):

    if tool_name and tool_name.lower() == "Lecture Material".lower():
        retriever = get_lecture_material_retrieval(embeddings_model)
        tools = create_retriever_tool(
            retriever,
            "Lecture-Material",
            """""",
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_separator="\n\n-------------------------------------\n\n"
        )

    elif tool_name and tool_name.lower() == "Seminar Material".lower():
        retriever = get_seminar_material_retrieval(embeddings_model)
        tools = create_retriever_tool(
            retriever,
            "Seminar-Material",
            """""",
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_separator="\n\n-------------------------------------\n\n"
        )

    elif tool_name and tool_name.lower() == "Organisational Material".lower():
        retriever = get_organisational_material_retrieval(embeddings_model)
        tools = create_retriever_tool(
            retriever,
            "Organisational-Material",
            """""",
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_separator="\n\n-------------------------------------\n\n"
        )

    else:
        retriever = get_all_material_retrieval(embeddings_model)
        tools = create_retriever_tool(
            retriever,
            "All-Material",
            """""",
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_separator="\n\n-------------------------------------\n\n"
        )

    return retriever, [tools]


def get_custom_additional_system_message():
    custom_additional_message = """\n\nImportant Instructions:\n
- Assistant must respond exclusively in German, using formal 'Sie/Ihr' address forms.
- Assistant should never respond in English or include file names or reference sources in responses.
- Assistant must break down the user's queries into sub-questions if the queries involve multiple topics.
- Assistant must base answers only on the user's input and provided documents without explicitly stating the answering process.
- Assistant should use tools to ensure accurate answers but may rely on previous chat history when applicable.
- Assistant must include the user's query in tool's input for completeness and context, assistant also never ignores any question from the user's query.
- Assistant must provide detailed and comprehensive responses, including all relevant information from the documents.
- Assistant must respond to yes/no questions with a definitive 'yes' or 'no', providing a succinct, justified explanation based solely on the documents.
- Assistant must state it does not know the answer if it is not found in the documents.
- Assistant must be extremely accurate and consistent with its response to the user's query.
- Assistant must ensure correctness and comprehensiveness in its responses and not repeat the user’s query.
- Assistant must make sure that its answer precisely corresponds to the user's query.
Let's think step by step.
"""

    return custom_additional_message


def condense_prompt(prompt: ChatPromptValue, llm) -> ChatPromptValue:
    messages = prompt.to_messages()
    num_tokens = llm.get_num_tokens_from_messages(messages)
    ai_function_messages = messages[3:]
    i = 0
    while num_tokens > 12_000:
        ai_function_message = ai_function_messages[i]
        context = ai_function_message.content.split("-------------------------------------")[:-1]
        ai_function_message.content = "-------------------------------------".join(context)
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:3] + ai_function_messages
        )
        i = (i + 1) % len(ai_function_messages)
    messages = messages[:3] + ai_function_messages
    return ChatPromptValue(messages=messages)


def get_executor(tool_name):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0, verbose=False)
    #llm = ChatGroq(temperature=0.0, model_name="llama-3.1-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
    #Best embedding model for mixtral / LLama 
    #Best practise for prompt
    #Smith.langchain
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    retriever, tools = get_retrieval(tool_name, embeddings_model)

    assistant_system_message = """You are a helpful assistant."""

    custom_additional_message = get_custom_additional_system_message()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", assistant_system_message + custom_additional_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True
    )

    agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | (lambda x: condense_prompt(x, llm))
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True,
                                   handle_parsing_errors=True, early_stopping_method="force", max_iterations=10,
                                   stream_runnable=False, verbose=False)


    return agent_executor, conversational_memory


def convert_memory_to_list(memory: ConversationBufferWindowMemory):
    buffer = []
    messages = memory.chat_memory.messages
    for mem in messages:
        if isinstance(mem, HumanMessage):
            buffer.append(HumanMessage(content=mem.content.split("Additional Instructions:")[0]))
        else:
            buffer.append(AIMessage(content=mem.content))
    return buffer


def chat_with_memory(input, agent_executor, conversational_memory, st_callback):
    if st_callback:
        out = agent_executor.invoke({
            "input": input,
            "chat_history": convert_memory_to_list(conversational_memory)
        }, {"callbacks": [st_callback]})
    else:
        out = agent_executor.invoke({
            "input": input,
            "chat_history": convert_memory_to_list(conversational_memory)
        })
    conversational_memory.chat_memory.add_user_message(input)
    conversational_memory.chat_memory.add_ai_message(out["output"])

    return out


def get_explanation(result):
    explanation = ""
    for step in result["intermediate_steps"]:
        tool_input = "Question: " + step[0].tool_input["query"]
        tool = "Tool: " + step[0].tool
        documents = "Documents: " + step[1]
        explanation += tool_input + "\n\n" + tool + "\n\n" + documents + "\n\n" + "*********************************" + "\n\n"

    return explanation


def add_cohere_costs(openai_callback, response):
    cohere_cost_per_search_unit = 0.001
    openai_callback.total_cost += len(response["intermediate_steps"]) * cohere_cost_per_search_unit

    return openai_callback


def generate_response(input, agent_executor, conversational_memory, st_callback):
    start_time = time.time()

    with get_openai_callback() as openai_callback:
        response = chat_with_memory(input, agent_executor, conversational_memory, st_callback)
        
        explanation = get_explanation(response)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to generate response: {time_taken:.4f} seconds")

    openai_callback = add_cohere_costs(openai_callback, response)

    return response, explanation, openai_callback

