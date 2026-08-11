import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class LlamaManager:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")

        self.client = Groq(
            api_key=api_key
        )
        self.model = "llama-3.3-70b-versatile"

    def generate_answer(self, question, selected_clauses):
        context = ""
        for clause in selected_clauses:
            context += f"--- Clause ID: {clause['clause_id']} (Doc: {clause['document']}) ---\n{clause['text']}\n\n"

        prompt = f"""
You are an expert Legal AI Analyst. 

Your task is to analyze ALL provided legal clauses below and produce a comprehensive, detailed legal report answering the user's question.

CRITICAL INSTRUCTIONS:
1. Examine EVERY clause provided in the context. Do not rely on just one clause.
2. Cross-reference related clauses to build a complete analysis.
3. Cite specific Clause IDs (e.g., [Clause 1], [Clause 3]) wherever relevant.
4. Structure the response professionally into clear sections:
   - Executive Summary
   - Detailed Legal Findings (Referencing all relevant clauses)
   - Synthesis & Conclusion
5. If the uploaded legal clauses do not contain sufficient information to answer the question, state explicitly:
   "The uploaded legal documents do not contain sufficient information to answer this question."

------------------------
RELEVANT LEGAL CLAUSES
------------------------
{context}

------------------------
USER QUESTION
------------------------
{question}

------------------------
LEGAL REPORT
------------------------
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content