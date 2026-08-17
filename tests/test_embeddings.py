from openai import OpenAI
import os
from dotenv import load_dotenv

from rag.embeddings import get_embedding


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


text = "Python is used for machine learning."


vector = get_embedding(
    client,
    text
)


print("Vector length:", len(vector))
print("First 10 values:", vector[:10])