import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class LlamaManager:

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            try:
                self.api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                self.api_key = None

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

        NOTE: Only a small number of clauses may be retrieved for any
        given question. Because we cannot change how many clauses are
        retrieved, this prompt instead asks the model to extract the
        maximum possible depth and detail out of whatever clauses it
        has been given, so the final report reads as substantive and
        complete rather than short.
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
Relevant Provisions, Detailed Analysis, Synthesis, or Conclusion,
unless the depth of the answer genuinely calls for section breaks.

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
DEPTH REQUIREMENT (APPLIES TO EVERY ANSWER TYPE)
============================================================

You have been given a limited number of clauses. Treat that as a reason
to go deeper on each one, not a reason to write a short answer.

For every clause you rely on, do not just restate what it says. Also
address, wherever the clause text supports it:

- What the clause is actually doing within the agreement (its function
  and purpose), not only its literal wording.
- Any conditions, triggers, deadlines, or prerequisites it sets out.
- The practical consequence for each party if the condition is met,
  and the consequence if it is not met.
- Any rights, obligations, restrictions, or discretion it creates for
  each named party (e.g. the firm vs. the client).
- How it connects to or depends on the other retrieved clauses (e.g.
  one clause referencing a fee or paragraph defined elsewhere).
- Anything left blank, undefined, or reserved for later completion in
  the clause (such as names, dates, or amounts), and what that implies
  practically.
- Any notable exceptions, carve-outs, or limits stated in the clause.

Expand each of these points into full sentences and connected
paragraphs rather than short fragments. Do not simply summarize in one
line per clause — walk through the reasoning the way a careful legal
analyst would when briefing a client, so that someone who has not read
the original document would still understand the mechanics, not just
the topic, of each clause.

1. FOLLOW THE USER'S INTENT

If the user asks for a SUMMARY or asks to "summarize":
- Provide a thorough, well-developed summary of the relevant document
  content, following the depth requirement above for each clause.
- Cover the important points, conditions, rights, duties, restrictions,
  procedures, exceptions, consequences, and other relevant information
  present in the retrieved clauses.
- Use connected paragraphs where appropriate, and close with a short
  paragraph tying the clauses together into an overall picture of what
  they establish (e.g. the framework, timing, or relationship they
  create).
- You may use a small number of headings or bullets when they genuinely
  improve readability, but prose should carry most of the explanation.
- Do not make the summary short. Prefer a fuller, more explanatory
  treatment over a terse one whenever the clauses support it.

If the user asks a SIMPLE FACTUAL QUESTION:
- Answer directly first.
- Then explain the reasoning behind the answer using the clause(s),
  including relevant conditions or exceptions, so the answer is
  complete rather than a bare fact.

If the user asks "WHAT DOES THIS MEAN?", "EXPLAIN", or asks to explain
a clause:
- Explain the relevant provision in clear, easy-to-understand language,
  following the depth requirement above.
- Include the important conditions and consequences.
- Do not merely repeat the clause word-for-word.
- Give enough detail that a non-lawyer would understand not just what
  the clause says, but why it matters and what happens as a result.

If the user asks for a DETAILED EXPLANATION or ANALYSIS:
- Give a comprehensive answer.
- Explain the relevant provisions and how they relate to the question
  and to each other.
- Distinguish explicit statements from reasonable interpretation.

If the user asks a YES/NO question:
- Start with Yes, No, or "The document does not provide enough
  information."
- Then explain why using the relevant clauses, in full paragraphs,
  following the depth requirement above.

If the user asks to COMPARE provisions:
- Compare the relevant clauses clearly and in detail.
- Explain similarities, differences, and how they interact.
- Use a table only if it genuinely improves the comparison.

If the user asks for a LIST, STEPS, CONDITIONS, RIGHTS, DUTIES,
or other enumerated information:
- Use bullets or numbered points because the user explicitly requested
  that format, but still give each bullet a full explanatory sentence
  or two rather than a fragment.

If the question is conversational or asks something specific:
- Respond naturally rather than using a predetermined template, while
  still applying the depth requirement above.

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
- Going deeper on the clauses you do have means explaining their
  mechanics and implications more fully -- it does NOT mean inventing
  facts that are not in the context.

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
- Do not repeat the same information verbatim, but you may return to a
  clause from a different angle (e.g. its purpose, then its effect)
  without that counting as repetition.
- Do not be unnecessarily brief. When in doubt, explain further rather
  than stopping early.
- Match the amount of detail to the user's request, but treat the
  depth requirement above as a floor, not a ceiling.

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
    # ESTIMATE TOKEN COUNT (ROUGH, NO EXTRA DEPENDENCIES)
    # ============================================================

    def _estimate_tokens(self, text):
        """
        Rough estimate of token count without pulling in a tokenizer
        dependency. English legal text averages roughly 4 characters
        per token, so this is a conservative approximation used only
        to keep requests inside the account's tokens-per-minute (TPM)
        budget -- it does not need to be exact.
        """

        if not text:
            return 0

        return max(
            1,
            len(text) // 4
        )


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

            system_message = (
                "You are a document-grounded "
                "legal question answering assistant. "
                "You must answer using only the "
                "retrieved legal document clauses. "
                "When only a few clauses are provided, "
                "explain each one thoroughly -- its "
                "purpose, conditions, consequences, and "
                "connections to the other clauses -- "
                "rather than giving a short summary."
            )

            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # ----------------------------------------------------
            # ACCOUNT TOKEN BUDGET (tokens per minute)
            # ----------------------------------------------------
            # The Groq "on_demand" service tier used by this account
            # currently enforces a combined prompt + completion limit
            # of 8000 tokens per minute (TPM). Setting a fixed
            # max_completion_tokens (e.g. 8000) ignores the size of
            # the prompt itself and triggers a 413 rate_limit_exceeded
            # error as soon as any real prompt is attached.
            #
            # Instead, estimate the prompt size first and only ask
            # for as much completion room as is left in the budget,
            # with a safety margin for estimation error.

            tpm_budget = int(
                os.getenv(
                    "GROQ_TPM_BUDGET",
                    "8000"
                )
            )

            safety_margin = 300

            estimated_prompt_tokens = (
                self._estimate_tokens(system_message)
                +
                self._estimate_tokens(prompt)
            )

            available_for_completion = (
                tpm_budget
                -
                estimated_prompt_tokens
                -
                safety_margin
            )

            # Always leave enough room to produce a real answer,
            # but never request more than the account can take.
            max_completion_tokens = max(
                256,
                min(
                    available_for_completion,
                    4000
                )
            )

            # ----------------------------------------------------
            # CALL GROQ, WITH ONE AUTOMATIC RETRY ON RATE LIMIT
            # ----------------------------------------------------
            # If the estimate above was still too generous (prompt
            # tokenized larger than our rough character-based guess),
            # Groq will return a 413 / rate_limit_exceeded error.
            # In that case, retry once with a smaller completion
            # budget instead of failing the whole request.

            attempts = [
                max_completion_tokens,
                max(256, max_completion_tokens // 2),
            ]

            last_error = None

            for attempt_tokens in attempts:

                try:

                    response = self.client.chat.completions.create(

                        model=self.model,

                        messages=messages,

                        temperature=0.2,

                        max_completion_tokens=attempt_tokens,

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

                    last_error = exc

                    error_text = str(exc).lower()

                    is_rate_limit = (
                        "rate_limit_exceeded" in error_text
                        or "413" in error_text
                        or "tokens per minute" in error_text
                    )

                    # Only retry for token/rate-limit style errors.
                    # Any other error (auth, network, etc.) should
                    # fail immediately instead of retrying.
                    if not is_rate_limit:
                        break

                    continue

            raise RuntimeError(
                f"Groq API request failed.\n"
                f"Model: {self.model}\n"
                f"Error: {last_error}"
            ) from last_error


        # ========================================================
        # LOCAL MODEL
        # ========================================================

        if self.mode == "local":

            try:

                result = self.generator(

                    prompt,

                    max_new_tokens=1200,

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