import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from preprocessing.pdf_extractor import PDFExtractor
from preprocessing.clause_chunker import ClauseChunker
from models.sentence_bert import SentenceBERT
from retrieval.retrieval import Retriever
from retrieval.dcbd import DCBD
from vectorstore.faiss_manager import FAISSManager
from llm.llama_manager import LlamaManager
from retrieval.relevance_validator import RelevanceValidator
from evaluation.faithfulness import FaithfulnessEvaluator
# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="LegalRAG AI",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------------
# Create Required Folders
# -------------------------------------------------------

os.makedirs("extracted_text", exist_ok=True)
os.makedirs("chunks", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# -------------------------------------------------------
# Initialize Modules
# -------------------------------------------------------

load_dotenv()

pdf_extractor = PDFExtractor()
chunker = ClauseChunker()
embedding_model = SentenceBERT()
retriever = Retriever()
dcbd = DCBD()
faiss_manager = FAISSManager()
validator = RelevanceValidator()

llama = None
llm_mode = None
try:
    llama = LlamaManager()
    llm_mode = getattr(llama, "mode", None)
    if llm_mode == "groq":
        st.success("LLM initialized with GROQ API key.")
    elif llm_mode == "local":
        st.success("LLM initialized with local transformers fallback.")
    else:
        st.warning(
            f"LLM initialization failed: {getattr(llama, 'error', 'unknown error')}. "
            "Install transformers & torch or set GROQ_API_KEY."
        )
except Exception as exc:
    st.warning(
        f"LLM initialization failed: {exc}. "
        "Install transformers & torch or set GROQ_API_KEY."
    )

faithfulness_evaluator = FaithfulnessEvaluator(
    embedding_model
)


def build_fallback_report(question, selected_clauses):
    header = (
        "### Legal Analysis Report (Fallback)\n"
        "_Generated from retrieved clauses only; no external LLM used._\n\n"
    )
    findings = []
    if not selected_clauses:
        return (
            header +
            "No supporting clauses were selected. Please upload relevant PDF documents and ask a question related to them."
        )

    for clause in selected_clauses:
        findings.append(
            f"Clause {clause['clause_id']} ({clause['document']}): {clause['text']}"
        )

    report = (
        header +
        "**Question:** " + question + "\n\n"
        "**Key clause summaries:**\n"
    )
    report += "\n\n".join(findings[:5])
    report += (
        "\n\n**Conclusion:** The answer above is derived directly from the retrieved legal clauses. "
        "If these clauses are insufficient to answer the question fully, consult the original documents or enable GROQ_API_KEY for an LLM-generated analysis."
    )
    return report


# -------------------------------------------------------
# UI
# -------------------------------------------------------

st.title("⚖️ LegalRAG AI")
st.subheader("Intelligent Legal Document Analysis using Sentence-BERT")

st.markdown("---")

uploaded_files = st.file_uploader(
    "📂 Upload Legal PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown("---")

if uploaded_files:

    st.header("📄 Uploaded Documents")

    for file in uploaded_files:

        st.success(file.name)

else:

    st.info("No documents uploaded.")

# -------------------------------------------------------
# Analyze Documents
# -------------------------------------------------------

if st.button("🚀 Analyze Documents"):

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

        st.stop()

    for uploaded_file in uploaded_files:

        st.markdown("---")

        st.header(f"📄 {uploaded_file.name}")

        # =====================================
        # STEP 1 : PDF Extraction
        # =====================================

        pdf_info = pdf_extractor.extract(uploaded_file)

        st.success("✅ PDF Extracted Successfully")

        st.write("Pages :", pdf_info["pages"])
        st.write("Characters :", pdf_info["characters"])

        st.text_area(
            "Preview",
            pdf_info["preview"],
            height=200
        )

        # =====================================
        # STEP 2 : Clause Chunking
        # =====================================

        document_name = uploaded_file.name.replace(".pdf", "")

        clauses = chunker.chunk_document(

            pdf_info["text"],

            document_name

        )

        st.success("✅ Clause Chunking Completed")

        st.write("Total Clauses :", len(clauses))

        st.subheader("First Three Clauses")

        for clause in clauses[:3]:

            with st.expander(

                f"Clause {clause['clause_id']}"

            ):

                st.write(clause["text"])

        # =====================================
        # STEP 3 : Sentence-BERT
        # =====================================

        chunk_file = os.path.join(

            "chunks",

            document_name + "_chunks.json"

        )

        embedding_info = embedding_model.generate(

            chunk_file

        )

        st.success("✅ Embeddings Generated")

        st.write(

            "Embedding Dimension :",

            embedding_info["dimension"]

        )

        st.write(

            "Total Embeddings :",

            embedding_info["total_chunks"]

        )

        st.code(

            embedding_info["embeddings"][0][:20]

        )

        # =====================================
        # STEP 4 : Build FAISS
        # =====================================

        embedding_file = embedding_info["embedding_file"]

        faiss_info = faiss_manager.build_index(

            embedding_file,

            chunk_file,

            document_name

        )

        st.success("✅ FAISS Index Created")

        st.write(

            "Vectors Stored :",

            faiss_info["vectors"]

        )

        st.write(

            "Embedding Dimension :",

            faiss_info["dimension"]

        )

        st.write(

            "Index File :",

            faiss_info["index_path"]

        )

st.markdown("---")

st.success("🎉 Knowledge Base Created Successfully!")

# =====================================================
# Ask Question
# =====================================================

st.markdown("---")

st.header("💬 Ask a Legal Question")

query = st.text_input(
    "Enter your question"
)

if st.button("🔍 Search"):

    if query.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload documents first.")
        st.stop()

    # ----------------------------------------
    # Retrieval
    # ----------------------------------------

    all_results = []

    for uploaded_file in uploaded_files:

        document_name = uploaded_file.name.replace(
            ".pdf",
            ""
        )

        index_path = os.path.join(
            "faiss_index",
            document_name + ".index"
        )

        metadata_path = os.path.join(
            "faiss_index",
            document_name + "_metadata.json"
        )

        results = retriever.search(
            query,
            index_path,
            metadata_path,
            top_k=5
        )

        all_results.extend(results)

    # ----------------------------------------
    # Sort Results
    # ----------------------------------------

    all_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_results = all_results[:10]

    # ----------------------------------------
    # Relevance Validation
    # ----------------------------------------

    valid, message = validator.validate(
        top_results
    )

    if not valid:

        st.error(
            "❌ The question is not related "
            "to the uploaded legal documents."
        )

        st.info(
            "Please ask a question based on "
            "the uploaded PDFs."
        )

        st.stop()

    # ----------------------------------------
    # Display Retrieved Clauses
    # ----------------------------------------

    st.subheader("📑 Retrieved Clauses")

    for item in top_results:

        with st.expander(
            f"Clause {item['clause_id']} | "
            f"Similarity: {item['score']:.4f}"
        ):

            st.write(item["text"])

    # ----------------------------------------
    # DCBD
    # ----------------------------------------

    selected = dcbd.select_clauses(
        top_results
    )

    st.subheader("⭐ DCBD Selected Clauses")

    for item in selected:

        st.write(
            f"Clause {item['clause_id']} | "
            f"Similarity: {item['score']:.4f}"
        )

        st.info(
            item["text"]
        )

        # ----------------------------------------
    # LLaMA
    # ----------------------------------------

    st.subheader("📑 Legal Analysis Report")

    report_generated = False
    report = ""

    if llama is None or llm_mode is None:
        st.warning(
            "No LLM is available. Install transformers & torch or set GROQ_API_KEY."
        )
        st.info(
            "To enable LLM-based report generation, either add GROQ_API_KEY to .env "
            "or install transformers and torch for local fallback."
        )
    else:
        with st.spinner("Generating legal report..."):
            report = llama.generate_answer(
                question=query,
                selected_clauses=selected
            )
            report_generated = True

    # ----------------------------------------
    # Display LLaMA Report
    # ----------------------------------------

    if report_generated:
        st.markdown(report)
    else:
        report = build_fallback_report(
            query,
            selected
        )
        st.markdown(report)

    # ----------------------------------------
    # Overall LLM Evaluation
    # ----------------------------------------

    if report_generated:
        evaluation = faithfulness_evaluator.evaluate(
            answer=report,
            selected_clauses=selected
        )

        faithfulness_score = float(
            evaluation.get(
                "faithfulness_score",
                0.0
            )
        )

        semantic_similarity = float(
            evaluation.get(
                "semantic_similarity",
                faithfulness_score / 100.0
            )
        )

        hallucination_rate = max(
            0.0,
            min(
                100.0 - faithfulness_score,
                100.0
            )
        )

        answer_quality = evaluation.get(
            "answer_quality",
            "Evaluation completed"
        )

        st.markdown("---")

        st.subheader(
            "🛡️ Overall LLM Answer Evaluation"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Faithfulness Score",
                f"{faithfulness_score:.2f}%"
            )

        with col2:
            st.metric(
                "Estimated Hallucination Risk",
                f"{hallucination_rate:.2f}%"
            )

        with col3:
            st.metric(
                "Answer-Context Similarity",
                f"{semantic_similarity:.4f}"
            )

        st.info(
            f"📊 Overall Assessment: {answer_quality}"
        )

        st.download_button(
            label="📥 Download Legal Report (.txt)",
            data=(
                "LEGAL ANALYSIS REPORT\n\n"
                f"Question:\n{query}\n\n"
                f"Answer:\n{report}\n\n"
                "--------------------------------\n"
                "OVERALL EVALUATION\n"
                "--------------------------------\n"
                f"Faithfulness Score: {faithfulness_score:.2f}%\n"
                f"Estimated Hallucination Risk: {hallucination_rate:.2f}%\n"
                f"Answer-Context Similarity: {semantic_similarity:.4f}\n"
                f"Overall Assessment: {answer_quality}\n"
            ),
            file_name="legal_analysis_report.txt",
            mime="text/plain"
        )
    else:
        st.markdown("---")
        st.info(
            "Legal evaluation is disabled until report generation is enabled."
        )