import logging
from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import JSONResponse
from typing import List
import os, time
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core.storage.docstore import SimpleDocumentStore
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import json
import nest_asyncio
nest_asyncio.apply()
# Configure logging
logging.basicConfig(
    filename='app.log',  # Specify the log file name
    filemode='a',         # Append mode; use 'w' to overwrite each time
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()
app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Setup environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TITLE = os.getenv("TITLE")
ROLE = os.getenv("ROLE").split(',')
PAGE = os.getenv("PAGE").split(",")
ASK_QUESTION = os.getenv("ASK_QUESTION")
ASK = os.getenv("ASK")
UPLOAD_DOC = os.getenv("UPLOAD_DOC")
E_QUESTION = os.getenv("E_QUESTION")
SECTION = os.getenv("SECTION").split(",")
DOCSTORE = os.getenv("DOCSTORE").split(",")
COLLECTION = os.getenv("COLLECTION").split(",")
DATABASE = os.getenv("DATABASE").split(",")
P_QUESTION = os.getenv("P_QUESTION")
INSERT_DOCUMENT = os.getenv("INSERT_DOCUMENT")
ADD_DOC = os.getenv("ADD_DOC")
DOC_ADDED = os.getenv("DOC_ADDED")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DELETE_DOC = os.getenv("DELETE_DOC")
C_DELETE = os.getenv("C_DELETE")
api_key = os.getenv("UNSTRUCTURED_API_KEY")
api_url = os.getenv("UNSTRUCTURED_API_URL")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOC_DELETED = os.getenv("DOC_DELETED")
N_DOC = os.getenv("N_DOC")
image = os.getenv("image")
imagess = os.getenv("imagess")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY
Settings.llm = OpenAI(model=LLM_MODEL) 
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)
CSV_FILE_PATH = "record_results.csv"

app.secret_key = "SECRET_KEY"  # Store secret key in .env for security

# Map the department to its index in the lists
DEPARTMENT_TO_INDEX = {
    "human_resources": 0,
    "legal": 1,
    "finance": 2,
    "operations": 3,
    "healthcare": 4,
    "insurance": 5,
    "learning_and_development": 6,
    "others": 7
}
class Session:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def __contains__(self, item):
        return item in self.data

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __iter__(self):
        return iter(self.data)
session = Session()
@app.get("/", response_class = HTMLResponse)
async def index(request: Request):
    """Main page."""
    try:
        logging.info('Rendering index page')
        return templates.TemplateResponse(request= request, name="index.html")
    except Exception as e:
        logging.error(f"Error loading index page: {e}")
        return JSONResponse({"status": "error", "message": "Error loading index page."})
def hybrid_retrieve(query, docstore, vector_index, bm25_retriever, alpha=0.5):
    """Perform hybrid retrieval using BM25 and vector-based retrieval."""
    # Get results from BM25
    try:
        bm25_results = bm25_retriever.retrieve(query)
        # Get results from the vector store
        vector_results = vector_index.as_retriever(similarity_top_k=2).retrieve(query)
    except Exception as e:
        logging.error(e)
        return JSONResponse("Error with retriever")
    # Combine results with weighting
    combined_results = {}
    # Weight BM25 results
    for result in bm25_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + (1 - alpha)
    # Weight vector results
    for result in vector_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + alpha

    # Sort results based on the combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    # Return the top N results
    return [docstore.get_document(doc_id) for doc_id, _ in sorted_results[:4]]



@app.route("/admin", methods=["GET", "POST"])
async def admin_page(request: Request):
    """Admin page to manage documents."""
    try:
        if request.method == "POST":
            form_data = await request.form()  # Parse the form data
            selected_section = form_data.get("section")
            
            # Ensure selected_section is valid
            if selected_section not in SECTION:
                raise ValueError("Invalid section selected.")
            
            collection_name = COLLECTION[SECTION.index(selected_section)]
            db_path = DATABASE[SECTION.index(selected_section)]
            
            logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")
            
            return templates.TemplateResponse(
                'admin.html',
                {
                    "request": request,
                    "section": selected_section,
                    "collection": collection_name,
                    "db_path": db_path
                }
            )

        logging.info('Rendering admin page')
        return templates.TemplateResponse(
            'admin.html',
            {
                "request": request,
                "sections": SECTION
            }
        )
    except Exception as e:
        logging.error(f"Error rendering admin page: {e}")
        return JSONResponse(
            {"status": "error", "message": f"Error rendering admin page: {str(e)}"},
            status_code=500
        )
@app.post("/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form(...),
    db_path: str = Form(...)
):
    """Handle file uploads for documents."""
    try:
        if files:
            logging.info(f"Handling upload for collection: {collection}, DB Path: {db_path}")
            for file in files:
                file_content = await file.read()
                file_name = file.filename
                
                try:
                    # Parse the uploaded file using LlamaParse
                    parsed_text = use_llamaparse(file_content, file_name)

                    # Split the parsed document into chunks
                    base_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
                    nodes = base_splitter.get_nodes_from_documents([Document(text=parsed_text)])

                    # Initialize storage context (defaults to in-memory)
                    storage_context = StorageContext.from_defaults()

                    # Prepare for storing document chunks
                    base_file_name = os.path.basename(file_name)
                    chunk_ids = []
                    metadatas = []

                    for i, node in enumerate(nodes):
                        chunk_id = f"{base_file_name}_{i + 1}"
                        chunk_ids.append(chunk_id)

                        metadata = {"type": base_file_name, "source": file_name}
                        metadatas.append(metadata)

                        document = Document(text=node.text, metadata=metadata, id_=chunk_id)
                        storage_context.docstore.add_documents([document])

                    # Persist storage context to a JSON file
                    collection_file = next((item for item in DOCSTORE if collection in item), None)
                    if collection_file and os.path.exists(collection_file):
                        with open(collection_file, "r") as f:
                            existing_documents = json.load(f)

                        # Update document store with new chunks
                        storage_context.docstore.persist(collection_file)
                        merged_dict = {doc: doc for doc in existing_documents["docstore/data"]}
                        final_dict = {"docstore/data": merged_dict}

                        with open(collection_file, "w") as f:
                            json.dump(final_dict, f, indent=4)
                    else:
                        storage_context.docstore.persist(collection_file)
                    collection_instance = init_chroma_collection(db_path, collection)

                    # Initialize vector store index and add document chunks to collection
                    embed_model = OpenAIEmbedding()
                    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
                    batch_size = 500
                    for i in range(0, len(nodes), batch_size):
                        batch_nodes = nodes[i : i + batch_size]
                        try:
                            collection_instance.add(
                                documents=[node.text for node in batch_nodes],
                                metadatas=metadatas[i : i + batch_size],
                                ids=chunk_ids[i : i + batch_size],
                            )
                            time.sleep(5)  # Add a retry with a delay

                        except:
                            # Handle rate limit by adding a delay or retry mechanism
                            print("Rate limit error has occurred at this moment")
                    # collection_instance = init_chroma_collection(db_path, collection)
                    # collection_instance.add(
                    #     documents=[node.text for node in nodes],
                    #     metadatas=metadatas,
                    #     ids=chunk_ids
                    # )
                    logging.info(f"Files uploaded and processed successfully for collection: {collection}")
                    return JSONResponse({"status": "success", "message": "Documents uploaded successfully."})
                
                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {e}")
                    return JSONResponse({"status": "error", "message": f"Error processing file {file_name}."})

        logging.warning("No files uploaded.")
        return JSONResponse({"status": "error", "message": "No files uploaded."})
    except Exception as e:
        logging.error(f"Error in upload_files: {e}")
        return JSONResponse({"status": "error", "message": "Error during file upload."})
def use_llamaparse(file_content, file_name):
    try:
        with open(file_name, "wb") as f:
            f.write(file_content)
        
        # Ensure the result_type is 'text', 'markdown', or 'json'
        parser = LlamaParse(result_type='text', verbose=True, language="en", num_workers=2)
        documents = parser.load_data([file_name])
        
        os.remove(file_name)
        
        res = ''
        for i in documents:
            res += i.text + " "
        return res
    except Exception as e:
        logging.error(f"Error parsing file: {e}")
        raise

def init_chroma_collection(db_path, collection_name):
    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_or_create_collection(collection_name, embedding_function=openai_ef)
        logging.info(f"Initialized Chroma collection: {collection_name} at {db_path}")
        return collection
    except Exception as e:
        logging.error(f"Error initializing Chroma collection: {e}")
        raise


@app.get("/documents")
async def show_documents(request: Request):
    """Show available documents."""
    try:
        # Retrieve query parameters
        collection_name = request.query_params.get("collection")
        db_path = request.query_params.get("db_path")

        if not collection_name or not db_path:
            raise ValueError("Missing 'collection' or 'db_path' query parameters.")

        # Initialize the collection
        collection = init_chroma_collection(db_path, collection_name)

        # Retrieve metadata and IDs from the collection
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']

        # Create a dictionary mapping document names to IDs
        doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                doc_name = meta['source'].split('\\')[-1]
                if doc_name not in doc_name_to_id:
                    doc_name_to_id[doc_name] = []
                doc_name_to_id[doc_name].append(doc_id)

        # Get the unique document names
        doc_list = list(doc_name_to_id.keys())

        # Logging the successful retrieval
        logging.info(f"Documents retrieved successfully for collection: {collection_name}")

        # Render the template with the document list
        return templates.TemplateResponse(
            'admin.html',
            {
                "request": request,
                "section": collection_name,
                "documents": doc_list,
                "collection": collection_name,
                "db_path": db_path,
                "sections": SECTION,
            }
        )

    except Exception as e:
        logging.error(f"Error showing documents: {e}")
        return JSONResponse({"status": "error", "message": "Error showing documents."})
@app.route("/delete_document", methods=["POST"])
async def delete_document(request: Request):
    """Handle document deletion."""
    try:
        doc_name = request.form.get("doc_name")
        collection_name = request.form.get("collection")
        db_path = request.form.get("db_path")
        
        ids_to_delete = session['doc_name_to_id'].get(doc_name, [])
        
        if ids_to_delete:
            collection = init_chroma_collection(db_path, collection_name)
            collection.delete(ids=ids_to_delete)
            
            # Update document store
            for i in range(len(DOCSTORE)):
                if collection_name in DOCSTORE[i]:
                    coll = DOCSTORE[i]
                    break
            if os.path.exists(coll):
                with open(coll, "r") as f:
                    existing_documents = json.load(f)

                # Remove deleted document from docstore
                merged_dict = {doc: doc for doc in existing_documents["docstore/data"] if doc not in ids_to_delete}
                final_dict = {"docstore/data": merged_dict}
                with open(coll, "w") as f:
                    json.dump(final_dict, f, indent=4)
            
            logging.info(f"Document '{doc_name}' deleted successfully.")
            return JSONResponse({"status": "success", "message": f"Document '{doc_name}' deleted successfully."})
        else:
            logging.warning(f"Document '{doc_name}' not found for deletion.")
            return JSONResponse({"status": "error", "message": "Document not found."})
    except Exception as e:
        logging.error(f"Error deleting document '{doc_name}': {e}")
        return JSONResponse({"status": "error", "message": "Error deleting document."})
@app.route("/query", methods=["GET", "POST"])
async def query_page(request: Request):
    """Handle document queries."""
    try:
        # Ensure chat_history is initialized
        if 'chat_history' not in session:
            session.set('chat_history', [])

        response_text = ""
        query_text = ""

        if request.method == "POST":
            form_data = await request.form()
            department = form_data.get("department")
            query_text = form_data.get("query_text")

            department_index = DEPARTMENT_TO_INDEX.get(department)
            if department_index is None:
                response_text = "Invalid department selected."
            else:
                docstore_file = DOCSTORE[department_index]
                collection_name = COLLECTION[department_index]
                db_path = DATABASE[department_index]

                if not os.path.exists(docstore_file):
                    response_text = f"Document store not found for {department}."
                else:
                    # Proceed with querying the documents
                    collection = init_chroma_collection(db_path, collection_name)

                    if "documents" in collection.get() and len(collection.get()['documents']) > 0:
                        vector_store = ChromaVectorStore(chroma_collection=collection)
                        docstore = SimpleDocumentStore.from_persist_path(docstore_file)
                        storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
                        vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=OpenAIEmbedding(api_key=OPENAI_API_KEY))

                        bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=2)
                        retrieved_nodes = hybrid_retrieve(query_text, docstore, vector_index, bm25_retriever, alpha=0.8)
                        context_str = "\n\n".join([node.get_content().replace('{', '').replace('}', '')[:4000] for node in retrieved_nodes])

                        qa_prompt_str = (
                            "Context information is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Given the context information and not prior knowledge, "
                            "answer the question: {query_str}\n"
                        )
                        fmt_qa_prompt = qa_prompt_str.format(context_str=context_str, query_str=query_text)

                        chat_text_qa_msgs = [
                            ChatMessage(
                                role=MessageRole.SYSTEM,
                                content="You are an AI language model designed to provide precise and contextually relevant responses."
                            ),
                            ChatMessage(
                                role=MessageRole.USER,
                                content=fmt_qa_prompt
                            ),
                        ]

                        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
                        result = vector_index.as_query_engine(text_qa_template=text_qa_template, llm=OpenAI(model=LLM_MODEL)).query(query_text)
                        response_text = result.response if result else "No response available."

                    # Save the chat history
                    session.get('chat_history').append({'role': 'user', 'text': query_text})
                    session.get('chat_history').append({'role': 'bot', 'text': response_text})

        # Render the query page and pass the chat history
        chat_history = session.get('chat_history', [])
        return templates.TemplateResponse('query.html', {"request": request, "query_text": query_text, "response_text": response_text, "chat_history": chat_history})
    except Exception as e:
        logging.error(f"Error in query page: {e}")
        return JSONResponse(
            {"status": "error", "message": f"Error in query page: {str(e)}"},
            status_code=500
        )

    
@app.get("/reset_chat")
async def reset_chat():
    """Clear chat history and redirect to the query page."""
    session.pop('chat_history', None)  # Clear chat history
    return RedirectResponse(url="/query", status_code=302)



