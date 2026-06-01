import streamlit as st
import pandas as pd
import os

from preprocessing import (
    preprocess,
    build_inverted_index,
    index_to_df,
    compare_stem_vs_lemma,
    get_stopwords,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IR System — Assignment 1",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --blue:       #2563EB;
    --blue-light: #EFF6FF;
    --blue-mid:   #BFDBFE;
    --dark:       #0F172A;
    --muted:      #64748B;
    --border:     #E2E8F0;
    --surface:    #F8FAFC;
    --white:      #FFFFFF;
    --green:      #16A34A;
    --green-bg:   #F0FDF4;
    --amber:      #D97706;
    --amber-bg:   #FFFBEB;
    --red:        #DC2626;
    --red-bg:     #FEF2F2;
    --radius:     10px;
    --shadow:     0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Global font ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stTextInput input {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    border-radius: var(--radius) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
    width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--dark) !important;
    box-shadow: var(--shadow) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Cards ── */
.ir-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.ir-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.ir-card-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--dark);
    line-height: 1.2;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--blue-light) !important;
    border: 2px dashed var(--blue-mid) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--blue) !important;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-blue   { background: var(--blue-light); color: var(--blue); }
.badge-green  { background: var(--green-bg);   color: var(--green); }
.badge-amber  { background: var(--amber-bg);   color: var(--amber); }
.badge-red    { background: var(--red-bg);     color: var(--red); }

/* ── Code / mono ── */
code, pre, .stCode {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Expanders ── */
details {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0 0.5rem !important;
    background: var(--surface) !important;
}

/* ── Info / warning / success boxes ── */
.ir-info    { background:var(--blue-light);  border-left:4px solid var(--blue);  color:#1E40AF; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }
.ir-success { background:var(--green-bg);    border-left:4px solid var(--green); color:#166534; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }
.ir-warn    { background:var(--amber-bg);    border-left:4px solid var(--amber); color:#92400E; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: var(--radius) !important;
    border-color: var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: var(--blue) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = {}  # {filename: content}
if "query" not in st.session_state:
    st.session_state.query = ""

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='padding:1.2rem 0 1rem;'>
        <div style='font-size:1.4rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.02em;'>
            🔍 IR System
        </div>
        <div style='font-size:0.75rem;color:#64748B;margin-top:2px;'>
            Assignment 1 — AIMLCZG537
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── File uploader in sidebar ──
    st.markdown(
        "<div style='font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;'>Upload Documents</div>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="upload",
        type=["txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload .txt or .csv files. Each file = one document.",
    )

    # Process uploaded files
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.documents:
                try:
                    content = f.read().decode("utf-8", errors="ignore")
                    st.session_state.documents[f.name] = content
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")

    # Load sample data button
    if st.button("Load sample docs", use_container_width=True):
        sample = {
            "doc1.txt": "Information retrieval is the activity of obtaining information resources relevant to an information need.",
            "doc2.txt": "A search engine is a software system designed to carry out web searches.",
            "doc3.txt": "Natural language processing is a subfield of linguistics and artificial intelligence.",
            "doc4.txt": "Stemming is the process of reducing inflected words to their word stem or root form.",
            "doc5.txt": "Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item.",
        }
        st.session_state.documents.update(sample)
        st.success(f"Loaded {len(sample)} sample documents")

    # Clear button
    if st.session_state.documents:
        if st.button("Clear all documents", use_container_width=True):
            st.session_state.documents = {}
            st.rerun()

    st.divider()

    # ── Query input ──
    st.markdown(
        "<div style='font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem;'>Search Query</div>",
        unsafe_allow_html=True,
    )
    st.session_state.query = st.text_input(
        "query",
        value=st.session_state.query,
        placeholder="e.g. information retrieval",
        label_visibility="collapsed",
    )

    # ── Retrieval mode ──
    st.markdown(
        "<div style='font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;margin:0.8rem 0 0.5rem;'>Retrieval Mode</div>",
        unsafe_allow_html=True,
    )
    retrieval_mode = st.selectbox(
        "mode",
        [
            "Inverted Index",
            "Biword Index",
            "Positional Index",
            "BST",
            "B-Tree",
            "Tolerant",
        ],
        label_visibility="collapsed",
    )

    # ── Preprocessing options ──
    st.markdown(
        "<div style='font-size:0.75rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:0.07em;margin:0.8rem 0 0.5rem;'>Preprocessing</div>",
        unsafe_allow_html=True,
    )
    opt_lower = st.checkbox("Lowercase", value=True)
    opt_stop = st.checkbox("Remove stop words", value=True)
    opt_stem = st.checkbox("Stemming", value=False)
    opt_lemma = st.checkbox("Lemmatization", value=False)

    st.divider()
    run_btn = st.button("🔍  Run Search", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
doc_count = len(st.session_state.documents)
total_words = sum(len(c.split()) for c in st.session_state.documents.values())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class='ir-card'>
        <div class='ir-card-title'>Documents</div>
        <div class='ir-card-value'>{doc_count}</div>
    </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class='ir-card'>
        <div class='ir-card-title'>Total words</div>
        <div class='ir-card-value'>{total_words:,}</div>
    </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class='ir-card'>
        <div class='ir-card-title'>Query</div>
        <div class='ir-card-value' style='font-size:1.1rem;padding-top:0.35rem;'>
            {st.session_state.query or '—'}
        </div>
    </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class='ir-card'>
        <div class='ir-card-title'>Mode</div>
        <div class='ir-card-value' style='font-size:1.1rem;padding-top:0.35rem;'>
            {retrieval_mode}
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📂  Upload & Preview",
        "🔤  Preprocessing",
        "🔍  Phrase Query",
        "🌳  Dictionary Search",
        "🧩  Tolerant Retrieval",
        "📝  Inference",
    ]
)

# ── TAB 1 — Upload & Preview ──────────────────
with tab1:
    st.markdown("### Document collection")

    if not st.session_state.documents:
        st.markdown(
            """<div class='ir-warn'>
            No documents loaded yet. Upload <code>.txt</code> or <code>.csv</code> files
            using the sidebar, or click <b>Load sample docs</b> to get started.
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class='ir-success'>
            ✔ &nbsp;{doc_count} document{'s' if doc_count != 1 else ''} loaded
            &nbsp;·&nbsp; {total_words:,} total words
        </div>""",
            unsafe_allow_html=True,
        )

        # Document table
        rows = []
        for name, content in st.session_state.documents.items():
            words = content.split()
            rows.append(
                {
                    "File": name,
                    "Words": len(words),
                    "Characters": len(content),
                    "Preview": content[:120].replace("\n", " ")
                    + ("…" if len(content) > 120 else ""),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Full document viewer
        st.markdown("#### View full document")
        selected = st.selectbox(
            "Select document", list(st.session_state.documents.keys())
        )
        if selected:
            with st.expander("Full content", expanded=True):
                st.text(st.session_state.documents[selected])

# ── TAB 2 — Preprocessing ─────────────────────
with tab2:
    st.markdown("### Text preprocessing pipeline")

    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>Upload documents first (Tab 1).</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='ir-info'>Select preprocessing options in the sidebar, then choose a document below.</div>",
            unsafe_allow_html=True,
        )

        doc_choice = st.selectbox(
            "Document to preprocess",
            list(st.session_state.documents.keys()),
            key="prep_doc",
        )
        raw_text = st.session_state.documents.get(doc_choice, "")

        result = preprocess(
            raw_text,
            do_lemma=opt_lemma,
            do_stem=opt_stem,
            lowercase=opt_lower,
            rm_stopwords=opt_stop,
        )

        st.markdown("#### Pipeline Stages")
        step_col1, step_col2 = st.columns(2)

        with step_col1:
            st.markdown("**① Raw Token**")
            st.code(result["raw_tokens"])

            st.markdown("**② After lowercase**")
            st.code(result["lower_tokens"])

        with step_col2:
            st.markdown("**③ After stop word removal**")
            removed = result["removed_sw"]
            st.code(result["clean_tokens"])
            if removed:
                st.markdown(f"Removed **{len(removed)}** stop words:")
                st.code(removed)
            st.markdown("**④ Final tokens**")
            st.code(result["final_tokens"])

        st.divider()
        # ─────────────────────────────────────────────
        #  Stem/Lemma map
        # ─────────────────────────────────────────────
        if result["stem_map"]:
            st.markdown("#### Stemming changes")
            stem_df = pd.DataFrame(
                result["stem_map"].items(), columns=["Orginals", "Stemmed"]
            )
            st.dataframe(stem_df, use_container_width=True, hide_index=True)

        if result["lemma_map"]:
            st.markdown("### Lemmatization changes")
            lemma_df = pd.DataFrame(
                result["lemma_map"].items(), columns=["Originals", "Lemma"]
            )
            st.dataframe(lemma_df, use_container_width=True, hide_index=True)

        st.divider()

        # ─────────────────────────────────────────────
        #  Inverted index
        # ─────────────────────────────────────────────
        st.markdown("#### Inverted index")
        with st.spinner("Building index ..."):
            index = build_inverted_index(
                st.session_state.documents,
                do_lemma=opt_lemma,
                do_stem=opt_stem,
                lowercase=opt_lower,
                rm_stopwords=opt_stop,
            )

        index_df = index_to_df(index)
        st.dataframe(index_df, use_container_width=True, hide_index=True)

        # ─────────────────────────────────────────────
        #  Stem vs Lemma comparison
        # ─────────────────────────────────────────────

        st.markdown("#### Stemming vs Lemmatization comparison")
        query_input = st.text_input(
            "Enter queries to compare (comma sperated)",
            value="information retrieval, search engine, word processing",
            key="compare_queries",
        )
        btn = st.button("Run comparison", key="comapre_btn")
        if btn:
            queries = [q.strip() for q in query_input.split(",") if q.split()]
            with st.spinner("Comparing ..."):
                comp_df = compare_stem_vs_lemma(st.session_state.documents, queries)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                better_counts = comp_df["Better"].value_counts()
                winner = better_counts.idxmax() if len(better_counts) else "Tie"
                st.markdown(
                    f"""<div class='ir-success'>
                ✔ &nbsp;<b>{winner}</b> performed better on average
                across {len(queries)} queries on this dataset.
            </div>""",
                    unsafe_allow_html=True,
                )

# ── TAB 3 — Phrase Query ──────────────────────
with tab3:
    st.markdown("### Phrase query — biword vs positional index")
    st.markdown(
        "<div class='ir-info'>Enter a phrase query in the sidebar to compare both indexes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='ir-card'>
        <div class='ir-card-title'>Coming soon</div>
        Implement <code>phrase_query.py</code> and call it here.
        Show biword results on the left and positional results on the right.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── TAB 4 — Dictionary Search ─────────────────
with tab4:
    st.markdown("### Dictionary search — BST vs B-Tree")
    st.markdown(
        "<div class='ir-info'>Timing benchmarks will appear here after implementing <code>tree_structures.py</code>.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='ir-card'>
        <div class='ir-card-title'>Results table (placeholder)</div>
        Populate with: Query | BST time (ms) | B-Tree time (ms) | Found
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── TAB 5 — Tolerant Retrieval ────────────────
with tab5:
    st.markdown("### Tolerant retrieval")
    st.markdown(
        "<div class='ir-info'>Wildcard, spelling correction, k-gram, and phonetic matching go here.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='ir-card'>
        <div class='ir-card-title'>Coming soon</div>
        Implement <code>tolerant.py</code> and display correction suggestions + matched documents.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── TAB 6 — Inference ─────────────────────────
with tab6:
    st.markdown("### Experimental results & inference")
    st.markdown(
        "<div class='ir-info'>Write your conclusions for each task here after running experiments.</div>",
        unsafe_allow_html=True,
    )

    questions = [
        (
            "Task B",
            "Which preprocessing technique improved retrieval quality the most?",
        ),
        ("Task B", "Was stemming or lemmatization better for your dataset? Why?"),
        (
            "Task C",
            "Which phrase query index was more accurate — biword or positional?",
        ),
        (
            "Task D",
            "Which tree structure was faster? Explain the time complexity difference.",
        ),
        (
            "Task E",
            "How effective was tolerant retrieval for misspelled and wildcard queries?",
        ),
        ("General", "What are the limitations of this system?"),
        ("General", "How can this system be improved in future?"),
    ]

    for tag, question in questions:
        badge_color = {
            "Task B": "blue",
            "Task C": "green",
            "Task D": "amber",
            "Task E": "blue",
            "General": "amber",
        }.get(tag, "blue")
        st.markdown(
            f"<span class='badge badge-{badge_color}'>{tag}</span>",
            unsafe_allow_html=True,
        )
        st.text_area(
            question,
            placeholder="Write your inference here...",
            key=f"inf_{tag}_{question[:20]}",
            height=80,
        )
        st.markdown("")
