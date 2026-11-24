import sys
sys.path.insert(0, "src")

from model_clients import query_model

print(query_model("gemini-3.5-flash-lite", "The code is 42.", "What is the code?"))