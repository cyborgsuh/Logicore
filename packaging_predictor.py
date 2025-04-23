import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the product dataset
product_df = pd.read_csv("Product_Dataset.csv")

# Format documents with all relevant features
product_df['combined'] = product_df.apply(lambda row: (
    f"Product ID: {row['Product_ID']}, Type: {row['Product_Type']}, "
    f"Weight: {row['Weight_kg']}kg, Fragile: {row['Fragile']}, "
    f"Temperature Condition: {row['Temp_Condition']}, Humidity Level: {row['Humidity_Level']}, "
    f"Recommended Packaging: {row['Packaging_Material']}"), axis=1)

# Use DataFrameLoader to turn rows into documents
loader = DataFrameLoader(product_df[['combined']], page_content_column="combined")
docs = loader.load()

# Embedding and Vectorstore
embedding = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding=embedding)

# Load Ollama model
llm = Ollama(model="llama3.2")

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a packaging expert AI. Use the given context to recommend appropriate packaging materials.
Context:
{context}

Question:
{question}

Answer:"""
        )
    }
)

# Load the test package
test_df = pd.read_csv("test_package.csv")

# Recommend for each test case
for index, row in test_df.iterrows():
    question = (f"What packaging should be used for a {row['Product_Type']} "
                f"that weighs {row['Weight_kg']}kg, is fragile: {row['Fragile']}, "
                f"requires {row['Temp_Condition']} temperature and "
                f"{row['Humidity_Level']} humidity level?")
    
    result = qa_chain.run(question)
    print(f"Product ID {row['Product_ID']} Recommendation: {result}")
