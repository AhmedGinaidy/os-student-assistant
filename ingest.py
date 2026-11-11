
from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Get the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FOLDER = BASE_DIR / "data" / "documents"
DB_FOLDER = BASE_DIR / "data" / "chroma_db"


def load_pdfs():
    documents = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return documents

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            for page in pages:
                page.metadata["document_name"] = pdf_file.name

            documents.extend(pages)
            print(f"Loaded: {pdf_file.name}")

        except Exception as error:
            print(f"Could not read {pdf_file.name}: {error}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"\nCreated {len(chunks)} text chunks.")

    return chunks


def save_to_database(chunks):
    if DB_FOLDER.exists():
        shutil.rmtree(DB_FOLDER)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_FOLDER)
    )

    print("\nDocuments saved successfully in ChromaDB!")


def main():
    print("Starting document processing...\n")

    documents = load_pdfs()

    if not documents:
        print("No documents were loaded. Check your PDF files.")
        return

    chunks = split_documents(documents)

    if not chunks:
        print("No text was extracted from the PDFs.")
        return

    save_to_database(chunks)

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    main()
