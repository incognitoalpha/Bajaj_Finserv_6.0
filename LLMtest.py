# reasoner_gemini.py

import os
import json
import google.generativeai as genai

# Configure Gemini API key (from environment or hardcoded)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Or replace with your actual key

# Prompt Builder: Combines user question, structured metadata, and retrieved clauses
def build_prompt(question: str, structured_data: dict, clauses: list) -> str:
    clause_text = "\n\n".join(
        [f"[{clause['clause_id']}] {clause['text']}" for clause in clauses]
    )

    return f"""
You are a legal and insurance document expert LLM.

User Question:
"{question}"

Extracted Context:
{json.dumps(structured_data, indent=2)}

Relevant Clauses:
{clause_text}

TASK:
- Determine if the answer to the question is found in the clauses.
- If yes, answer it precisely and concisely.
- Reference specific clause(s) for justification.
- Answer must be short but complete.

Return ONLY this JSON:
{{
  "answer": "<plain text response>",
  "justification": "<why this answer is correct>",
  "relevant_clauses": ["clause_id_1", "clause_id_2"]
}}
"""

# Main reasoning function
def generate_answer_with_gemini(question: str, structured_query: dict, retrieved_clauses: list) -> dict:
    try:
        # Prepare prompt
        prompt = build_prompt(question, structured_query, retrieved_clauses)

        # Load Gemini model
        model = genai.GenerativeModel("gemini-2.5-pro")

        # Send to LLM
        response = model.generate_content(prompt)

        # Parse response (expecting valid JSON)
        return json.loads(response.text)

    except Exception as e:
        return {
            "answer": "error",
            "justification": f"LLM error: {str(e)}",
            "relevant_clauses": []
        }
