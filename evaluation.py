import os
import dotenv
from langchain_community.chat_models import ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from ragas import evaluate as ragas_evaluate

from ragas.evaluation import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness, 
    AnswerRelevancy, 
    ContextRelevance, 
    ContextUtilization,
    ContextRecall
)

from agent_handler import get_llm

dotenv.load_dotenv()

os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HF_TOKEN")

# Choose an open-source LLM from Hugging Face Hub
# You might need to experiment with different models for optimal results
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Initialize HuggingFaceEndpoint for the Chat model
# llm = ChatHuggingFace(model_name=llm_model_name)
llm = get_llm(model_name=llm_model_name)

# Choose an open-source embedding model from Hugging Face Hub
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

# Initialize HuggingFaceEmbeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name=embedding_model_name
)

# Sample documents
documents = [
    Document(page_content="The Eiffel Tower is in Paris.", metadata={"source": "fact1"}),
    Document(page_content="Paris is the capital of France.", metadata={"source": "fact2"}),
    Document(page_content="London is the capital of England.", metadata={"source": "fact3"})
]

db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})

# Define a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

query = "Where is the Eiffel Tower?"
result = qa_chain.invoke({"query": query})

print(result)
'''
generated_answer = result["result"]
retrieved_docs = result["source_documents"]

no_ground_truth_metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    # ContextRecall(),
    ContextRelevance(),
    ContextUtilization()
]

# Evaluate using RAGAS (without ground truth)
ragas_result = ragas_evaluate(
    dataset=EvaluationDataset([
        SingleTurnSample(
            user_input=query,
            response=generated_answer,
            retrieved_contexts=[doc.page_content for doc in retrieved_docs],
        )
    ]),
    metrics=no_ground_truth_metrics
)
print("RAGAS scores (without ground truth):", ragas_result)
'''