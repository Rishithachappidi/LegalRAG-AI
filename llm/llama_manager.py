import os

from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


class LlamaManager:

    def __init__(self):

        # ====================================================
        # INITIAL STATE
        # ====================================================

        self.api_key = os.getenv("GROQ_API_KEY")

        self.mode = None
        self.error = None

        self.client = None
        self.model = None

        self.tokenizer = None
        self.generator = None

        # ====================================================
        # GROQ
        # ====================================================

        if self.api_key:

            try:

                from groq import Groq

                self.client = Groq(
                    api_key=self.api_key
                )

                # Current Groq model
                self.model = "openai/gpt-oss-120b"

                self.mode = "groq"

                return

            except Exception as exc:

                self.error = (
                    f"Failed to initialize GROQ client: {exc}"
                )

                self.mode = None

        else:

            self.error = (
                "GROQ_API_KEY is not set. "
                "Please add GROQ_API_KEY to your .env file."
            )

        # ====================================================
        # LOCAL LLM FALLBACK
        # ====================================================

        try:

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline
            )

            import torch

        except ImportError:

            return

        model_name = os.getenv(
            "LOCAL_LLM_MODEL",
            "distilgpt2"
        )

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name
            )

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=(
                    0
                    if torch.cuda.is_available()
                    else -1
                )
            )

            self.mode = "local"

            self.error = None

        except Exception as exc:

            self.error = (
                f"Failed to initialize local transformer "
                f"model '{model_name}': {exc}"
            )


    # ============================================================
    # BUILD QUESTION-AWARE LEGAL PROMPT
    # ============================================================

    def _build_prompt(
        self,
        question,
        selected_clauses
    ):
        """
        Build a question-adaptive prompt.

        The response style is determined by the user's question:
        - summarize / summary -> detailed prose summary
        - explain / meaning -> clear explanation
        - simple factual question -> direct concise answer
        - compare -> comparison
        - list / enumerate -> list only when requested
        - detailed / analyze -> detailed analysis
        - otherwise -> natural answer based on the question

        All factual/legal content must come only from the retrieved clauses.
        """

        if not selected_clauses:
            return f"""
You are a Legal Document Question Answering AI.

The user asked:
{question}

No relevant clauses were retrieved from the uploaded document.

Tell the user clearly that the available document context does not
contain enough information to answer the question.

Do not use outside legal knowledge.
Do not invent information.
"""

        context_parts = []

        for clause in selected_clauses:
            clause_id = clause.get("clause_id", "Unknown")
            document = clause.get("document", "Unknown Document")
            text = clause.get("text", "")

            context_parts.append(
                f"""
--- Clause ID: {clause_id} ---
Document: {document}

{text}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a document-grounded Legal Question Answering AI.

Your job is to answer the USER QUESTION naturally and appropriately
using ONLY the RETRIEVED LEGAL DOCUMENT CONTEXT below.

Do NOT give every answer in bullet points.
Do NOT force every answer into a fixed report structure.
Do NOT automatically create sections such as Executive Summary,
Relevant Provisions, Detailed Analysis, Synthesis, or Conclusion.

Instead, understand what the user is asking and choose the response
style that best fits the question.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED LEGAL DOCUMENT CONTEXT
============================================================

{context}

============================================================
HOW TO ANSWER
============================================================

1. FOLLOW THE USER'S INTENT

If the user asks for a SUMMARY or asks to "summarize":
- Provide a reasonably detailed summary of the relevant document content.
- Cover the important points, conditions, rights, duties, restrictions,
  procedures, exceptions, consequences, and other relevant information
  present in the retrieved clauses.
- Use connected paragraphs where appropriate.
- You may use a small number of headings or bullets when they genuinely
  improve readability.
- Do not make the summary unnecessarily short.

If the user asks a SIMPLE FACTUAL QUESTION:
- Answer directly.
- Keep the answer concise but complete.
- Explain the answer briefly when useful.
- Do not produce a long report.

If the user asks "WHAT DOES THIS MEAN?", "EXPLAIN", or asks to explain
a clause:
- Explain the relevant provision in clear, easy-to-understand language.
- Include the important conditions and consequences.
- Do not merely repeat the clause word-for-word.
- Give enough detail to make the provision understandable.

If the user asks for a DETAILED EXPLANATION or ANALYSIS:
- Give a more comprehensive answer.
- Explain the relevant provisions and how they relate to the question.
- Distinguish explicit statements from reasonable interpretation.

If the user asks a YES/NO question:
- Start with Yes, No, or "The document does not provide enough
  information."
- Then explain why using the relevant clauses.

If the user asks to COMPARE provisions:
- Compare the relevant clauses clearly.
- Explain similarities, differences, and how they interact.
- Use a table only if it genuinely improves the comparison.

If the user asks for a LIST, STEPS, CONDITIONS, RIGHTS, DUTIES,
or other enumerated information:
- Use bullets or numbered points because the user explicitly requested
  that format.

If the question is conversational or asks something specific:
- Respond naturally rather than using a predetermined template.

============================================================
DOCUMENT-GROUNDING RULES
============================================================

- Use ONLY the retrieved document context.
- Do NOT use outside legal knowledge.
- Do NOT invent laws, rules, penalties, dates, names, rights,
  obligations, exceptions, procedures, or facts.
- If the context does not contain enough information, say so clearly.
- Do not pretend that an inference is explicitly stated in the document.
- You may explain an inference, but clearly indicate that it is an
  interpretation based on the provided text.

============================================================
CLAUSE CITATIONS
============================================================

Support important legal statements with the actual Clause ID.

Use citations such as:
[Clause 1]
[Clause 3]
[Clause 2] and [Clause 5]

Only use Clause IDs that actually appear in the retrieved context.
Never invent a Clause ID.

============================================================
STYLE
============================================================

- Write naturally and professionally.
- Prefer paragraphs for explanations and summaries.
- Use bullets only when they improve the answer or the user asks for them.
- Do not repeat the same information.
- Do not be unnecessarily verbose for simple questions.
- Do not be unnecessarily brief for summaries or detailed questions.
- Match the amount of detail to the user's request.

============================================================
LEGAL DISCLAIMER
============================================================

You are a document-grounded legal information assistant.
You are not a lawyer.
Do not claim that your response constitutes professional legal advice.

============================================================
FINAL TASK
============================================================

Answer the USER QUESTION now.

LEGAL ANSWER:
"""

        return prompt


    # ============================================================
    # GENERATE ANSWER
    # ============================================================

    def generate_answer(
        self,
        question,
        selected_clauses
    ):

        # --------------------------------------------------------
        # Build question-aware prompt
        # --------------------------------------------------------

        prompt = self._build_prompt(
            question,
            selected_clauses
        )

        # ========================================================
        # GROQ
        # ========================================================

        if self.mode == "groq":

            try:

                response = self.client.chat.completions.create(

                    model=self.model,

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document-grounded "
                                "legal question answering assistant. "
                                "You must answer using only the "
                                "retrieved legal document clauses."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.1,

                    max_completion_tokens=4000,

                    top_p=0.9,

                    stream=False
                )

                # ------------------------------------------------
                # Safely extract response
                # ------------------------------------------------

                if (
                    response.choices
                    and response.choices[0].message
                ):

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    if answer:

                        return answer.strip()


                return (
                    "The language model returned an empty answer."
                )


            except Exception as exc:

                raise RuntimeError(
                    f"Groq API request failed.\n"
                    f"Model: {self.model}\n"
                    f"Error: {exc}"
                ) from exc


        # ========================================================
        # LOCAL MODEL
        # ========================================================

        if self.mode == "local":

            try:

                result = self.generator(

                    prompt,

                    max_new_tokens=700,

                    do_sample=True,

                    temperature=0.7,

                    top_p=0.9,

                    num_return_sequences=1
                )

                generated_text = result[0][
                    "generated_text"
                ]


                # ------------------------------------------------
                # Remove prompt from generated result
                # ------------------------------------------------

                if generated_text.startswith(prompt):

                    generated_text = generated_text[
                        len(prompt):
                    ]


                return generated_text.strip()


            except Exception as exc:

                raise RuntimeError(
                    f"Local LLM generation failed: {exc}"
                ) from exc


        # ========================================================
        # NO LLM AVAILABLE
        # ========================================================

        raise RuntimeError(
            self.error or
            "No LLM is available. "
            "Please configure GROQ_API_KEY."
        )