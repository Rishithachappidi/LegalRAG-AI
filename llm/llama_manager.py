import os


class LlamaManager:

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        self.mode = None
        self.error = None

        # -----------------------------------------
        # GROQ
        # -----------------------------------------

        if self.api_key:

            try:

                from groq import Groq

                self.client = Groq(
                    api_key=self.api_key
                )

                self.mode = "groq"

                self.model = "llama-3.1-8b-instant"

                return

            except Exception as exc:

                self.error = (
                    f"Failed to initialize GROQ client: {exc}"
                )

        # -----------------------------------------
        # LOCAL LLM
        # -----------------------------------------

        try:

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline
            )

            import torch

        except ImportError:

            self.error = (
                "GROQ_API_KEY is not set and local "
                "transformers are unavailable. "
                "Install transformers and torch, "
                "or set GROQ_API_KEY."
            )

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

        except Exception as exc:

            self.error = (
                f"Failed to initialize local transformer "
                f"model '{model_name}': {exc}"
            )

    # =====================================================
    # BUILD LEGAL PROMPT
    # =====================================================

    def _build_prompt(
        self,
        question,
        selected_clauses
    ):

        context = ""

        for clause in selected_clauses:

            context += (
                f"--- Clause ID: "
                f"{clause['clause_id']} "
                f"(Document: {clause['document']}) ---\n"
                f"{clause['text']}\n\n"
            )

        prompt = f"""
You are an expert Legal AI Analyst.

You must answer the user's legal question using ONLY
the legal clauses provided in the context.

Do not invent legal provisions, facts, dates, penalties,
rights, obligations, or interpretations that are not
supported by the provided clauses.

Analyze ALL provided clauses and cross-reference them
where necessary.

USER QUESTION:
{question}

RELEVANT LEGAL CLAUSES:
{context}

Prepare a COMPLETE and DETAILED legal analysis.

Use the following structure:

1. Executive Summary
Provide a direct answer to the user's question.

2. Relevant Legal Provisions
Identify the clauses that are relevant to the question.
Cite their Clause IDs.

3. Detailed Legal Analysis
Explain what each relevant clause means in relation to
the user's question.
Cross-reference clauses where they are connected.

4. Synthesis
Combine the relevant provisions and explain how they
work together.

5. Limitations
Clearly mention if the provided documents do not contain
enough information to answer any part of the question.

6. Conclusion
Provide a clear final conclusion based only on the
uploaded legal documents.

IMPORTANT:
- Examine every provided clause.
- Do not ignore relevant clauses.
- Cite Clause IDs such as [Clause 1] or [Clause 3].
- Do not use outside legal knowledge.
- Do not fabricate information.
- If the documents do not contain enough information,
  explicitly state that.

LEGAL ANALYSIS:
"""

        return prompt

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate_answer(
        self,
        question,
        selected_clauses
    ):

        prompt = self._build_prompt(
            question,
            selected_clauses
        )

        # -----------------------------------------
        # GROQ
        # -----------------------------------------

        if self.mode == "groq":

            response = self.client.chat.completions.create(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.1,

                max_completion_tokens=1000,

                top_p=0.9
            )

            return response.choices[0].message.content

        # -----------------------------------------
        # LOCAL MODEL
        # -----------------------------------------

        if self.mode == "local":

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

            # Remove prompt from generated result
            if generated_text.startswith(prompt):

                generated_text = generated_text[
                    len(prompt):
                ]

            return generated_text.strip()

        raise ValueError(
            self.error or
            "Unsupported LLM mode."
        )