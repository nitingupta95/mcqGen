import os
import json
import pandas as pd # type: ignore
import traceback
from dotenv import load_dotenv # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore # ✅ Updated import
 
load_dotenv()
 
KEY = os.getenv("OPENAI_API_KEY")

if not KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in a .env file.")
 
llm = ChatOpenAI(
    openai_api_key=KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.3
)
