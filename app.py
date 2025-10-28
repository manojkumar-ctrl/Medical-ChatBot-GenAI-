from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone  # ✅ Changed from PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone as PineconeClient

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Check if API keys are available
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set. Please add it to your .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please add it to your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------
# Initialize Pinecone
# ---------------------------
pc = PineconeClient(api_key=PINECONE_API_KEY)

# ---------------------------
# Initialize embeddings
# ---------------------------
embeddings = download_hugging_face_embeddings()

# ---------------------------
# Connect to Pinecone index
# ---------------------------
index_name = "medicalbot"

# Get the Pinecone index
pinecone_index = pc.Index(index_name)

# Create vector store
docsearch = Pinecone(  # ✅ Changed from PineconeVectorStore
    index=pinecone_index,
    embedding=embeddings,
    text_key="text"
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ---------------------------
# Initialize LLM (Google Gemini)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.4,
    max_output_tokens=300
)

# ---------------------------
# Setup RetrievalQA chain using modern LangChain approach
# ---------------------------
from langchain_core.prompts import ChatPromptTemplate

print("Setting up chain...")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "Context: {context}\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("Creating qa_chain...")
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Chain setup complete!")

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/test", methods=["GET"])
def test():
    return "Flask app is working!"

@app.route("/get", methods=["POST"])
def chat():
    print("=== NEW REQUEST ===")
    print("Request method:", request.method)
    print("Request form:", request.form)
    
    if "msg" not in request.form:
        print("No 'msg' in request form!")
        return "Error: No message provided"
    
    msg = request.form["msg"]
    print("User input:", msg)
    
    try:
        print("Invoking chain...")
        response = qa_chain.invoke(msg)
        print("Raw response:", repr(response))
        print("Response type:", type(response))
        print("Response length:", len(str(response)) if response else 0)
        
        if not response or str(response).strip() == "":
            print("Empty response detected!")
            return "Sorry, I couldn't generate a response. Please try again."
        
        print("Returning response:", str(response)[:100] + "..." if len(str(response)) > 100 else str(response))
        return str(response)
    except Exception as e:
        print("Error occurred:", str(e))
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)