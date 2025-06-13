# Import necessary modules:
# Flask for building the web application
from flask import Flask, render_template, request, jsonify
# CORS for handling Cross-Origin Resource Sharing (important for frontend/backend communication)
from flask_cors import CORS
# secure_filename for safely handling uploaded file names
from werkzeug.utils import secure_filename
# os for interacting with the operating system (like creating folders, handling file paths)
import os
# shutil for high-level file operations (though not directly used in the current version, often useful)
import shutil

# --- LlamaIndex Imports ---
# Settings: This is where I configure my LLM and embedding model globally for LlamaIndex
from llama_index.core.settings import Settings
# VectorStoreIndex: This is the core data structure for my RAG system, where documents are indexed
# SimpleDirectoryReader: This helps me load documents from a specified folder
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# HuggingFaceEmbedding: This allows me to use a local, free embedding model from Hugging Face
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Ollama: This is the specific connector to use local LLMs running via Ollama
from llama_index.llms.ollama import Ollama

# --- Flask App Setup ---
# Initialize my Flask application
app = Flask(__name__)
# Enable CORS for my app, so my frontend (even on a different port/origin) can talk to it
CORS(app)
# Define the folder where I'll store uploaded documents
app.config['UPLOAD_FOLDER'] = 'uploaded_docs'
# Make sure this upload folder exists. If not, create it.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Variables ---
# I'll use this to store my LlamaIndex after a document is uploaded and processed.
# It needs to be global so my '/ask' route can access it.
global_index = None
# I'll use this to keep track of the path to the currently uploaded document.
# This helps with cleaning up old files later.
global_document_path = None

# --- Routes ---

@app.route('/')
def index():
    """
    This function handles requests to the root URL (e.g., http://127.0.0.1:5000/).
    It simply renders my main HTML page (e.g., index.html) to the user.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    This function handles file uploads from the frontend.
    It takes the uploaded document, saves it, and then processes it to create a LlamaIndex.
    """
    # I need to declare these global to modify them
    global global_index
    global global_document_path

    # Check if a file was actually sent in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # Check if the file field was empty (no file selected)
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Sanitize the filename to prevent security issues (e.g., directory traversal attacks)
    filename = secure_filename(file.filename)
    # Create the full path where the file will be saved
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # --- Cleanup Previous Document and Index ---
    # Before saving a new file, I want to clear out any old documents
    # and reset the index to ensure I'm always working with the latest uploaded content.
    # If there was a previous document, delete it from the folder.
    if global_document_path and os.path.exists(global_document_path):
        os.remove(global_document_path)
    # Also, remove any other files that might be in the upload folder
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        if os.path.isfile(full_path):
            os.remove(full_path)

    # Save the newly uploaded file to the designated upload folder
    file.save(filepath)
    # Store its path in my global variable for future reference/cleanup
    global_document_path = filepath

    try:
        # --- LlamaIndex Document Processing Pipeline ---
        # 1. Load the document(s) from my upload folder.
        # SimpleDirectoryReader will automatically detect and parse supported file types.
        documents = SimpleDirectoryReader(input_dir=app.config['UPLOAD_FOLDER']).load_data()

        # 2. Configure the Embedding Model:
        # This model converts text into numerical vectors (embeddings)
        # that capture semantic meaning. 'all-MiniLM-L6-v2' is a good, small, free model.
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 3. Configure the LLM (Large Language Model):
        # This is the model that will answer questions. I'm using Ollama to run a local model.
        # I need to make sure 'llama2' (or 'phi3:mini' or 'tinyllama' etc.) is downloaded via Ollama.
        # The request_timeout is set high because local LLMs can sometimes be slow to respond.
        Settings.llm = Ollama(model="llama2", request_timeout=360.0)

        # 4. Build the Vector Store Index:
        # This step creates the searchable index from my documents.
        # It uses the embed_model to create embeddings and the LLM (implicitly) for overall context.
        global_index = VectorStoreIndex.from_documents(documents)

        # If everything worked, send a success message back to the frontend.
        return jsonify({'message': f'File "{filename}" uploaded and processed successfully.'})

    except Exception as e:
        # --- Error Handling During Processing ---
        # If anything goes wrong during document loading, embedding, or indexing:
        # 1. Clean up the partially uploaded file to avoid clutter.
        if os.path.exists(filepath):
            os.remove(filepath)
        # 2. Reset my global variables to indicate no document is loaded/indexed.
        global_document_path = None
        global_index = None
        # 3. Print the error to my console for debugging.
        print(f"Error during file upload and processing: {e}")
        # 4. Send an error message back to the frontend.
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    This function handles user questions.
    It takes a question from the frontend and uses the LlamaIndex to find and generate an answer.
    """
    # Need to access my global index
    global global_index
    # Get the JSON data from the request body (which should contain the 'question')
    data = request.get_json()
    question = data.get('question')

    # Basic validation: Check if a question was actually provided
    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    # Check if a document has been uploaded and indexed yet
    if global_index is None:
        return jsonify({'error': 'No file uploaded yet. Please upload a document first.'}), 400

    try:
        # --- Querying the LlamaIndex ---
        # 1. Create a query engine from my global index.
        # This engine knows how to retrieve relevant info from the index and
        # then pass it to the configured LLM to formulate an answer.
        query_engine = global_index.as_query_engine()
        # 2. Query the engine with the user's question.
        response = query_engine.query(question)
        # 3. Send the generated answer back to the frontend.
        return jsonify({'answer': str(response)})

    except Exception as e:
        # --- Error Handling During Querying ---
        # If any error occurs while trying to answer the question:
        # 1. Print the error to the console.
        print(f"Error during question query: {e}")
        # 2. Send an error message back to the frontend.
        return jsonify({'error': f'Error querying document: {str(e)}'}), 500

# --- Main Execution Block ---
# This ensures the Flask app runs only when the script is executed directly (not imported as a module).
if __name__ == '__main__':
    # Set environment variables for Flask.
    # FLASK_APP tells Flask where to find my application.
    os.environ['FLASK_APP'] = 'app.py'
    # FLASK_ENV set to 'development' enables debug mode (auto-reloads on code changes, shows detailed errors).
    os.environ['FLASK_ENV'] = 'development'
    # Run the Flask app. debug=True means it runs in development mode.
    # It will typically run on http://127.0.0.1:5000/
    app.run(debug=True)