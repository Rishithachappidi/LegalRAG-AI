import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
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

# =========================================================
# DARK THEME POLISH (CSS)
# =========================================================
# The base dark theme itself comes from .streamlit/config.toml
# (base="dark"), which makes every native Streamlit widget
# (buttons, metrics, alerts, tables, expanders, inputs) use
# correct dark-mode colors automatically. This CSS only adds
# extra visual polish on top of that -- custom fonts, card
# borders, and accent colors -- instead of fighting the theme
# with manual color overrides (which is what caused the
# white-text-on-white-background bug before).
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700;900&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Merriweather', serif !important;
        color: #f3e7bd !important;
        font-weight: 700 !important;
    }

    /* Subtle glowing gradient background instead of flat black */
    .stApp {
        background: radial-gradient(circle at top left, #141a2b 0%, #0d0f16 55%);
    }

    /* ===================================================
       REAL BORDERED CONTAINERS (st.container(border=True))
       -- the correct way to visually group widgets in
       Streamlit, unlike raw HTML <div> wrappers injected via
       st.markdown() which don't actually nest content.
    =================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161a26 !important;
        border: 1px solid #262c3d !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        padding: 8px 10px;
    }

    /* ===================================================
       BUTTONS
    =================================================== */

    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #c9a227 100%) !important;
        color: #0d0f16 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.6em 1.6em !important;
        transition: all 0.18s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(212, 175, 55, 0.4);
        filter: brightness(1.08);
    }

    /* ===================================================
       METRICS
    =================================================== */

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #1b2032 0%, #161a26 100%);
        border: 1px solid #262c3d;
        border-left: 4px solid #d4af37;
        border-radius: 10px;
        padding: 14px 16px;
    }

    div[data-testid="stMetricValue"] {
        color: #f3e7bd !important;
        font-weight: 700 !important;
    }

    /* ===================================================
       EXPANDERS
    =================================================== */

    [data-testid="stExpander"] {
        border: 1px solid #262c3d !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #f3e7bd !important;
    }

    /* ===================================================
       FILE UPLOADER
    =================================================== */

    [data-testid="stFileUploaderDropzone"] {
        background-color: #161a26 !important;
        border: 2px dashed #d4af37 !important;
        border-radius: 12px !important;
    }

    /* ===================================================
       TABLES
    =================================================== */

    .stTable th {
        background-color: #d4af37 !important;
        color: #0d0f16 !important;
        font-weight: 700 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
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


@st.cache_resource
def get_faithfulness_evaluator(_embedding_model):
    # Cached so the evaluator's underlying model is loaded ONCE
    # per app session, instead of being rebuilt on every
    # Streamlit rerun (every button click).
    return FaithfulnessEvaluator(_embedding_model)


faithfulness_evaluator = get_faithfulness_evaluator(
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


# =========================================================
# CHART HELPERS (dark-theme styled)
# =========================================================

def _band_color(value):
    """Semantic color for a 0-100 score band, chosen to stay
    vivid and readable against a dark background."""
    if value >= 85:
        return "#34d399"   # green
    elif value >= 70:
        return "#fbbf24"   # amber
    else:
        return "#f87171"   # red


def make_gauge_chart(value, title, accent_color):
    """Modern gauge chart (0-100) styled for a dark background."""

    number_color = _band_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 36, "family": "Merriweather", "color": number_color}
            },
            title={
                "text": f"<b>{title}</b>",
                "font": {"size": 15, "family": "Inter", "color": "#eef1f7"}
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#3a4257",
                    "tickfont": {"size": 10, "color": "#9aa3b8"}
                },
                "bar": {"color": accent_color, "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#2a1a1e"},
                    {"range": [50, 70], "color": "#2a2618"},
                    {"range": [70, 100], "color": "#17281f"}
                ],
                "threshold": {
                    "line": {"color": "#d4af37", "width": 3},
                    "thickness": 0.82,
                    "value": 70
                }
            }
        )
    )

    fig.update_layout(
        height=270,
        margin=dict(l=25, r=25, t=65, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#eef1f7"}
    )

    return fig


def make_comparison_chart(methods, faithfulness_scores, hallucination_scores):
    """Grouped bar chart comparing Faithfulness vs Hallucination
    Risk across the three detection methods, styled dark."""

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Faithfulness Score",
            x=methods,
            y=faithfulness_scores,
            marker=dict(color="#d4af37", cornerradius=8, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in faithfulness_scores],
            textposition="outside",
            textfont=dict(color="#f3e7bd", size=13, family="Inter"),
            hovertemplate="%{x}<br>Faithfulness: %{y:.2f}%<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            name="Hallucination Risk",
            x=methods,
            y=hallucination_scores,
            marker=dict(color="#f87171", cornerradius=8, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in hallucination_scores],
            textposition="outside",
            textfont=dict(color="#f87171", size=13, family="Inter"),
            hovertemplate="%{x}<br>Hallucination Risk: %{y:.2f}%<extra></extra>"
        )
    )

    fig.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.12,
        height=430,
        yaxis=dict(
            range=[0, 115],
            title="Percentage (%)",
            gridcolor="#262c3d",
            zeroline=False,
            color="#9aa3b8"
        ),
        xaxis=dict(
            tickfont=dict(size=13, family="Inter", color="#eef1f7"),
            color="#9aa3b8"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="Inter", color="#eef1f7")
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#eef1f7"}
    )

    return fig


# -------------------------------------------------------
# UI
# -------------------------------------------------------

st.title("⚖️ LegalRAG AI")
st.subheader("Intelligent Legal Document Analysis using Sentence-BERT")

st.markdown("---")

with st.container(border=True):

    uploaded_files = st.file_uploader(
        "📂 Upload Legal PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.header("📄 Uploaded Documents")

        for file in uploaded_files:

            st.success(file.name)

    else:

        st.info("No documents uploaded.")

st.markdown("---")

# -------------------------------------------------------
# Analyze Documents
# -------------------------------------------------------

if st.button("🚀 Analyze Documents"):

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

        st.stop()

    for uploaded_file in uploaded_files:

        with st.container(border=True):

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

                with st.expander(f"Clause {clause['clause_id']}"):

                    st.write(clause["text"])

            # =====================================
            # STEP 3 : Sentence-BERT
            # =====================================

            chunk_file = os.path.join(
                "chunks",
                document_name + "_chunks.json"
            )

            embedding_info = embedding_model.generate(chunk_file)

            st.success("✅ Embeddings Generated")

            st.write("Embedding Dimension :", embedding_info["dimension"])
            st.write("Total Embeddings :", embedding_info["total_chunks"])

            st.code(embedding_info["embeddings"][0][:20])

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

            st.write("Vectors Stored :", faiss_info["vectors"])
            st.write("Embedding Dimension :", faiss_info["dimension"])
            st.write("Index File :", faiss_info["index_path"])

    st.markdown("---")

    st.success("🎉 Knowledge Base Created Successfully!")

# =====================================================
# Ask Question
# =====================================================

st.markdown("---")

with st.container(border=True):

    st.header("💬 Ask a Legal Question")

    query = st.text_input(
        "Enter your question"
    )

    search_clicked = st.button("🔍 Search")

if search_clicked:

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

        document_name = uploaded_file.name.replace(".pdf", "")

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

    all_results.sort(key=lambda x: x["score"], reverse=True)

    top_results = all_results[:10]

    # ----------------------------------------
    # Relevance Validation
    # ----------------------------------------

    valid, message = validator.validate(top_results)

    if not valid:

        st.error("❌ The question is not related to the uploaded legal documents.")

        st.info("Please ask a question based on the uploaded PDFs.")

        st.stop()

    # ----------------------------------------
    # Display Retrieved Clauses
    # ----------------------------------------

    with st.container(border=True):

        st.subheader("📑 Retrieved Clauses")

        for item in top_results:

            with st.expander(
                f"Clause {item['clause_id']} | Similarity: {item['score']:.4f}"
            ):

                st.write(item["text"])

    # ----------------------------------------
    # DCBD
    # ----------------------------------------

    selected = dcbd.select_clauses(top_results)

    with st.container(border=True):

        st.subheader("⭐ DCBD Selected Clauses")

        for item in selected:

            st.write(f"Clause {item['clause_id']} | Similarity: {item['score']:.4f}")

            st.info(item["text"])

    # ----------------------------------------
    # LLaMA
    # ----------------------------------------

    with st.container(border=True):

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

        if report_generated:
            st.markdown(report)
        else:
            report = build_fallback_report(query, selected)
            st.markdown(report)

    # ----------------------------------------
    # Overall LLM Evaluation
    # ----------------------------------------

    if report_generated:

        with st.spinner(
            "Running hallucination detection "
            "(first run may take a little longer "
            "while the model loads)..."
        ):
            evaluation = faithfulness_evaluator.evaluate(
                answer=report,
                selected_clauses=selected
            )

        # ----------------------------------------------------
        # METHOD 1: Cosine Similarity (embedding-based)
        # ----------------------------------------------------

        faithfulness_score = float(evaluation.get("faithfulness_score", 0.0))

        semantic_similarity = float(
            evaluation.get("semantic_similarity", faithfulness_score / 100.0)
        )

        cosine_hallucination_rate = float(
            evaluation.get("hallucination_rate", 100.0 - faithfulness_score)
        )

        # ----------------------------------------------------
        # METHOD 2: BERTScore
        # ----------------------------------------------------

        bert_precision = float(evaluation.get("bert_precision", 0.0))
        bert_recall = float(evaluation.get("bert_recall", 0.0))
        bert_f1 = float(evaluation.get("bert_f1", 0.0))
        bert_score_pct = bert_f1 * 100

        bert_hallucination_rate = float(
            evaluation.get("bert_hallucination_rate", 100.0 - bert_score_pct)
        )

        # ----------------------------------------------------
        # METHOD 3: Jaccard Similarity
        # ----------------------------------------------------

        jaccard_similarity = float(evaluation.get("jaccard_similarity", 0.0))
        jaccard_score_pct = jaccard_similarity * 100

        jaccard_hallucination_rate = float(
            evaluation.get("jaccard_hallucination_rate", 100.0 - jaccard_score_pct)
        )

        answer_quality = evaluation.get("answer_quality", "Evaluation completed")

        with st.container(border=True):

            st.subheader("🛡️ Hallucination Detection — 3 Method Comparison")

            st.caption(
                "Each method independently estimates how well the "
                "generated answer is grounded in the retrieved clauses. "
                "Compare Faithfulness/Similarity (higher = better) "
                "against Hallucination Risk (lower = better) across "
                "methods below."
            )

            # -------------------------------------------------
            # GAUGE CHARTS — one per method
            # -------------------------------------------------

            g1, g2, g3 = st.columns(3)

            with g1:
                st.plotly_chart(
                    make_gauge_chart(
                        faithfulness_score,
                        "Cosine Similarity",
                        "#60a5fa"
                    ),
                    use_container_width=True
                )
                st.caption(f"Raw similarity: {semantic_similarity:.4f}")

            with g2:
                st.plotly_chart(
                    make_gauge_chart(
                        bert_score_pct,
                        "BERTScore (F1)",
                        "#c084fc"
                    ),
                    use_container_width=True
                )
                st.caption(
                    f"Precision: {bert_precision:.4f} | Recall: {bert_recall:.4f}"
                )

            with g3:
                st.plotly_chart(
                    make_gauge_chart(
                        jaccard_score_pct,
                        "Jaccard Similarity",
                        "#2dd4bf"
                    ),
                    use_container_width=True
                )
                st.caption(f"Raw Jaccard: {jaccard_similarity:.4f}")

            # -------------------------------------------------
            # PER-METHOD METRICS (exact numbers)
            # -------------------------------------------------

            st.markdown("#### 📌 Exact Scores")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Cosine — Faithfulness", f"{faithfulness_score:.2f}%")
                st.metric("Cosine — Hallucination Risk", f"{cosine_hallucination_rate:.2f}%")

            with c2:
                st.metric("BERTScore — Faithfulness (F1)", f"{bert_score_pct:.2f}%")
                st.metric("BERTScore — Hallucination Risk", f"{bert_hallucination_rate:.2f}%")

            with c3:
                st.metric("Jaccard — Faithfulness", f"{jaccard_score_pct:.2f}%")
                st.metric("Jaccard — Hallucination Risk", f"{jaccard_hallucination_rate:.2f}%")

            # -------------------------------------------------
            # COMPARISON BAR CHART
            # -------------------------------------------------

            st.markdown("#### 📊 Method Comparison")

            comparison_fig = make_comparison_chart(
                methods=["Cosine Similarity", "BERTScore", "Jaccard Similarity"],
                faithfulness_scores=[
                    faithfulness_score,
                    bert_score_pct,
                    jaccard_score_pct
                ],
                hallucination_scores=[
                    cosine_hallucination_rate,
                    bert_hallucination_rate,
                    jaccard_hallucination_rate
                ]
            )

            st.plotly_chart(comparison_fig, use_container_width=True)

            # -------------------------------------------------
            # SUMMARY TABLE
            # -------------------------------------------------

            st.markdown("#### 📋 Side-by-Side Summary")

            st.table(
                {
                    "Method": [
                        "Cosine Similarity",
                        "BERTScore",
                        "Jaccard Similarity"
                    ],
                    "Faithfulness Score (%)": [
                        f"{faithfulness_score:.2f}",
                        f"{bert_score_pct:.2f}",
                        f"{jaccard_score_pct:.2f}"
                    ],
                    "Hallucination Risk (%)": [
                        f"{cosine_hallucination_rate:.2f}",
                        f"{bert_hallucination_rate:.2f}",
                        f"{jaccard_hallucination_rate:.2f}"
                    ]
                }
            )

            if "excellent" in answer_quality.lower() or "good" in answer_quality.lower():
                st.success(f"📊 Overall Assessment: {answer_quality}")
            elif "moderate" in answer_quality.lower():
                st.warning(f"📊 Overall Assessment: {answer_quality}")
            else:
                st.error(f"📊 Overall Assessment: {answer_quality}")

        report_text = (
            "LEGAL ANALYSIS REPORT\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{report}\n\n"
            "--------------------------------\n"
            "HALLUCINATION DETECTION - 3 METHOD COMPARISON\n"
            "--------------------------------\n"
            "Method 1: Cosine Similarity (Embedding-Based)\n"
            f"  Faithfulness Score: {faithfulness_score:.2f}%\n"
            f"  Hallucination Risk: {cosine_hallucination_rate:.2f}%\n"
            f"  Raw Cosine Similarity: {semantic_similarity:.4f}\n\n"
            "Method 2: BERTScore\n"
            f"  Faithfulness Score (F1): {bert_score_pct:.2f}%\n"
            f"  Hallucination Risk: {bert_hallucination_rate:.2f}%\n"
            f"  Precision: {bert_precision:.4f}\n"
            f"  Recall: {bert_recall:.4f}\n\n"
            "Method 3: Jaccard Similarity\n"
            f"  Faithfulness Score: {jaccard_score_pct:.2f}%\n"
            f"  Hallucination Risk: {jaccard_hallucination_rate:.2f}%\n\n"
            f"Overall Assessment: {answer_quality}\n"
        )

        st.download_button(
            label="📥 Download Legal Report (.txt)",
            data=report_text,
            file_name="legal_analysis_report.txt",
            mime="text/plain"
        )

    else:
        st.markdown("---")
        st.info("Legal evaluation is disabled until report generation is enabled.")
