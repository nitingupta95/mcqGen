import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Load API key from .env
load_dotenv()
KEY = os.getenv("GOOGLE_API_KEY")
if not KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file.")

# LLM setup with Gemini
llm = ChatGoogleGenerativeAI(
    google_api_key=KEY,
    model="models/gemini-1.5-flash",
    temperature=0.3
)

# Example JSON template
RESPONSE_JSON = {
    "1": {"mcq": "multiple choice question", "options": {"a": "choice", "b": "choice", "c": "choice", "d": "choice"}, "correct": "correct answer"},
    "2": {"mcq": "multiple choice question", "options": {"a": "choice", "b": "choice", "c": "choice", "d": "choice"}, "correct": "correct answer"},
    "3": {"mcq": "multiple choice question", "options": {"a": "choice", "b": "choice", "c": "choice", "d": "choice"}, "correct": "correct answer"}
}

# Prompt 1 – updated to enforce strict JSON
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template="""
Text: {text}
You are an expert MCQ maker. Given the above text, create a quiz of {number} MCQs for {subject} students in {tone} tone.
Make sure:
- The questions are not repeated
- All questions conform strictly to the text
- The response is ONLY valid JSON — do NOT add explanations, code fences, or extra text.

Use the following JSON format exactly:
{response_json}
"""
)

# Prompt 2
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template="""
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students:
Evaluate the complexity in max 50 words. If needed, update questions to match cognitive level.
Quiz_MCQs:
{quiz}
"""
)

# Read text file
file_path = "/home/nitin/harikart/GenAI/MCQGenerator/data.txt"
with open(file_path, 'r') as file:
    TEXT = file.read()

# Build chains
quiz_chain = quiz_generation_prompt | llm | StrOutputParser()
review_chain = quiz_evaluation_prompt | llm | StrOutputParser()

# Inputs
NUMBER = 3
SUBJECT = "Machine Learning"
TONE = "hard"

with get_openai_callback() as cb:
    # Step 1: Generate quiz
    quiz_output = quiz_chain.invoke({
        "text": TEXT,
        "number": NUMBER,
        "subject": SUBJECT,
        "tone": TONE,
        "response_json": json.dumps(RESPONSE_JSON, indent=4)
    })

    # Step 2: Review quiz
    review_output = review_chain.invoke({
        "subject": SUBJECT,
        "quiz": quiz_output
    })

    print("Tokens used:", cb.total_tokens)

# Clean and parse JSON
try:
    clean_quiz = quiz_output.strip()

    # If Gemini still wraps with ```json ... ```
    if clean_quiz.startswith("```json"):
        clean_quiz = clean_quiz[len("```json"):].strip()
    if clean_quiz.endswith("```"):
        clean_quiz = clean_quiz[:-3].strip()

    # Remove anything after the last closing brace
    clean_quiz = clean_quiz[:clean_quiz.rfind('}')+1]

    quiz_dict = json.loads(clean_quiz)

except json.JSONDecodeError:
    quiz_dict = {"error": "Quiz output was not valid JSON", "raw": quiz_output}

# Display clean JSON output
print("\nGenerated Quiz (JSON):\n", json.dumps(quiz_dict, indent=4))
print("\nReview:\n", review_output)

 
if "error" not in quiz_dict:
    df = pd.DataFrame.from_dict(quiz_dict, orient="index")
    print("\nGenerated Quiz (Table View):\n")
    print(df)
else:
    print("Quiz could not be converted to DataFrame due to JSON error.")
    
    
df.to_csv("machine_learning.csv", index=False)