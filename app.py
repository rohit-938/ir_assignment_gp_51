import streamlit as st
import pandas as pd
import os

from preprocessing import (
    preprocess,
    build_inverted_index,
    index_to_df,
    compare_stem_vs_lemma,
)
from phrase_query import (
    build_biword_index,
    build_positional_index,
    query_biword,
    query_postional,
    biwords_index_to_dataframe,
    positional_index_to_dataframe,
    compare_indexes,
    false_positive_demo,
)
from indexing import (
    build_inverted_index as build_full_index,
    index_to_dataframe as full_index_df,
    index_statistics,
    stats_to_dataframe,
    parse_boolena_query,
    compute_tfidf,
)
from tree_structures import (
    build_bst_from_vocab,
    build_btree_from_vocab,
    extract_vocabulary,
    benchmark,
    tree_stats,
)
from tolerant import (
    build_kgram_index,
    query_wildcard,
    spelling_correction,
    spelling_correction_table,
    build_phonetic_index,
    phonetic_correction,
    phonetic_index_to_dataframe,
    kgram_index_to_dataframe,
    tolerant_search,
    _get_vocabulary,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
:root {
    --blue:#2563EB; --blue-light:#EFF6FF; --blue-mid:#BFDBFE;
    --dark:#0F172A; --muted:#64748B; --border:#E2E8F0;
    --surface:#F8FAFC; --white:#FFFFFF;
    --green:#16A34A; --green-bg:#F0FDF4;
    --amber:#D97706; --amber-bg:#FFFBEB;
    --red:#DC2626;   --red-bg:#FEF2F2;
    --radius:10px;
    --shadow:0 1px 3px rgba(0,0,0,0.08),0 1px 2px rgba(0,0,0,0.04);
}
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif !important; }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ padding:1.5rem 2rem !important; max-width:1200px !important; }
section[data-testid="stSidebar"]{ background:var(--dark) !important; border-right:none !important; }
section[data-testid="stSidebar"] *{ color:#CBD5E1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stTextInput label{
    color:#94A3B8 !important; font-size:0.78rem !important;
    letter-spacing:0.06em !important; text-transform:uppercase !important; font-weight:500 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]>div,
section[data-testid="stSidebar"] .stTextInput input{
    background:#1E293B !important; border:1px solid #334155 !important;
    color:#F1F5F9 !important; border-radius:var(--radius) !important;
}
section[data-testid="stSidebar"] .stButton>button{
    background:var(--blue) !important; color:white !important; border:none !important;
    border-radius:var(--radius) !important; font-weight:600 !important;
    width:100% !important; padding:0.6rem 1rem !important;
    letter-spacing:0.02em !important; transition:opacity 0.2s !important;
}
section[data-testid="stSidebar"] .stButton>button:hover{ opacity:0.88 !important; }
.stTabs [data-baseweb="tab-list"]{
    gap:4px; background:var(--surface); border-radius:var(--radius);
    padding:4px; border:1px solid var(--border); width:fit-content;
}
.stTabs [data-baseweb="tab"]{
    border-radius:7px !important; padding:0.4rem 1rem !important;
    font-size:0.85rem !important; font-weight:500 !important;
    color:var(--muted) !important; background:transparent !important;
    border:none !important; transition:all 0.15s !important;
}
.stTabs [aria-selected="true"]{
    background:var(--white) !important; color:var(--dark) !important; box-shadow:var(--shadow) !important;
}
.stTabs [data-baseweb="tab-border"]{ display:none !important; }
.ir-card{
    background:var(--white); border:1px solid var(--border);
    border-radius:var(--radius); padding:1.25rem 1.5rem;
    box-shadow:var(--shadow); margin-bottom:1rem;
}
.ir-card-title{ font-size:0.75rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.4rem; }
.ir-card-value{ font-size:1.8rem; font-weight:600; color:var(--dark); line-height:1.2; }
[data-testid="stFileUploader"]{
    background:var(--blue-light) !important; border:2px dashed var(--blue-mid) !important;
    border-radius:var(--radius) !important; padding:1rem !important; transition:border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover{ border-color:var(--blue) !important; }
[data-testid="stFileUploaderDropzone"]{ background:transparent !important; }
[data-testid="stFileUploaderDropzone"] *{ color:var(--blue) !important; }
.badge{ display:inline-block; padding:0.2rem 0.65rem; border-radius:999px; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; }
.badge-blue  { background:var(--blue-light); color:var(--blue); }
.badge-green { background:var(--green-bg);   color:var(--green); }
.badge-amber { background:var(--amber-bg);   color:var(--amber); }
.badge-red   { background:var(--red-bg);     color:var(--red); }
code,pre,.stCode{ font-family:'DM Mono',monospace !important; font-size:0.82rem !important; }
[data-testid="stDataFrame"]{ border:1px solid var(--border) !important; border-radius:var(--radius) !important; overflow:hidden !important; }
details{ border:1px solid var(--border) !important; border-radius:var(--radius) !important; padding:0 0.5rem !important; background:var(--surface) !important; }
.ir-info   { background:var(--blue-light); border-left:4px solid var(--blue);  color:#1E40AF; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }
.ir-success{ background:var(--green-bg);   border-left:4px solid var(--green); color:#166534; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }
.ir-warn   { background:var(--amber-bg);   border-left:4px solid var(--amber); color:#92400E; padding:0.75rem 1rem; border-radius:0 var(--radius) var(--radius) 0; font-size:0.88rem; margin:0.5rem 0; }
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div{
    border-radius:var(--radius) !important; border-color:var(--border) !important;
    font-family:'DM Sans',sans-serif !important;
}
.stTextInput input:focus{ border-color:var(--blue) !important; box-shadow:0 0 0 3px rgba(37,99,235,0.12) !important; }
.stButton>button[kind="primary"]{ background:var(--blue) !important; border:none !important; border-radius:var(--radius) !important; font-weight:600 !important; transition:opacity 0.2s !important; }
.stButton>button[kind="primary"]:hover{ opacity:0.88 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "documents": {},
    "query": "",
    "bw_index": None,
    "pos_index": None,
    "inv_index": None,
    "bst": None,
    "btree": None,
    "vocab": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

text3 = """
Misinformation of COVID-19 vaccines and vaccine hesitancy

The current study examined various types of misinformation related to the COVID-19 vaccines and their relationships to vaccine hesitancy and refusal. Study 1 asked a sample of full-time working professionals in the US (n = 505) about possible misinformation they were exposed to related to the COVID-19 vaccines. Study 2 utilized an online survey to examine U.S. college students’ (n = 441) knowledge about COVID-19 vaccines, and its associations with vaccine hesitancy and behavioral intention to get a COVID-19 vaccine.

Analysis of open-ended responses in Study 1 revealed that 57.6% reported being exposed to conspiratorial misinformation such as COVID-19 vaccines being harmful and dangerous. The results of a structural equation modeling analysis for Study 2 supported hypotheses predicting a negative association between knowledge level and vaccine hesitancy and between vaccine hesitancy and behavioral intention. Vaccine hesitancy mediated the relationship between vaccine knowledge and behavioral intention.

Findings across these studies suggest that exposure to misinformation and believing it to be true could increase vaccine hesitancy and reduce behavioral intention to get vaccinated.

Health misinformation can kill people, both directly and indirectly. During a public health crisis such as the COVID-19 pandemic, exposure to misinformation about the virus’ spread, symptoms of infection, testing opportunities, and prevention methods can lead to erroneous appraisals of the threat, maladaptive coping behaviors, and a range of fatal consequences.

More critically, misinformation about the new COVID-19 vaccines and their development process has the potential to induce high levels of vaccine hesitancy in the public, preventing vaccination rates sufficient for achieving herd immunity. Due to the high level of uncertainty caused by the pandemic and the relatively fast speed of vaccine development compared to traditional vaccines, the public naturally sought out information to address vaccine concerns and guide critical decision-making such as whether to get vaccinated.

However, separating relevant and valid information from false and distorted misinformation about COVID-19 vaccines is difficult when a vast amount of material is being conveyed through media outlets and websites of varying reliability and accuracy. One critically important challenge to obtaining reliable and accurate COVID-19 vaccine information includes the pervasive, unsolicited, and dubious pseudo-news items communicated via online and social media platforms.

Because many people acquire and share news via social media, misinformation can spread quickly through social networks, and the likelihood of exposure to disinformation and misrepresentations about vaccines from unverified sources is high. The resulting increase in public anxiety and negative emotional and behavioral responses complicates the process of advising the public through health experts and agencies such as the CDC and WHO.

The current research focuses on people’s perceptions of the nature and types of misinformation about COVID-19 vaccines. Additionally, this research examines the relationship between knowledge about COVID-19 vaccines, including relevant misinformation, and vaccine hesitancy and refusal.

By acknowledging the serious negative impact of health misinformation and its spread, research in infodemiology identifies the knowledge translation gap between evidence produced by experts and the public’s actual practices and beliefs. Research has identified various quality markers and their relations with outcome variables necessary for effective health communication on the internet.

Studies have shown that both algorithmic-based correction on Facebook and social correction via anonymous commenters can be effective in reducing beliefs in health misinformation. However, people who believe in conspiracy theories tend to discredit algorithm-based corrections.

More research is needed to:
(a) observe and define trends and prevalence of health misinformation on social media,
(b) understand how misinformation is shared,
(c) evaluate the reach and influence of misinformation, and
(d) develop and test effective interventions.
"""

text4 = """
HUMAN VACCINES & IMMUNOTHERAPEUTICS

Confidence in COVID-19 vaccine effectiveness and safety and its effect on vaccine uptake

COVID-19 is a major public health threat associated with increased disease burden, mortality, and economic loss to countries and communities. Safe and efficacious COVID-19 vaccines are key in halting and reversing the pandemic. Low confidence in vaccines has been one of the factors leading to hesitancy.

We aimed to assess COVID-19 vaccine confidence (safety and effectiveness), associated factors, and its effects on vaccine uptake among general community members in Tanzania. This was a community-based cross-sectional survey conducted from December 2021 to April 2022 in six regions of Tanzania mainland and two regions in Zanzibar.

Participants were interviewed using an electronic questionnaire. Multiple logistic regression models estimated odds ratios (ORs) and 95% confidence intervals (CI) for factors associated with vaccine confidence. All analyses were performed using SPSS version 25.0.

The study enrolled 3,470 Tanzanian community members. Their mean age was 40.3 years (SD ±14.9), and 34% were males. The proportion of COVID-19 vaccine confidence was 54.6%.

Geographical region, residence area, COVID-19 disease risk perception, and good knowledge of COVID-19 vaccines were significantly associated with vaccine confidence. Confidence in COVID-19 vaccines was associated with more than three times higher odds of vaccine uptake.

The findings indicate that confidence in COVID-19 vaccines was low in Tanzania. Innovative community engagement strategies and region-specific interventions are needed to improve knowledge and address community perceptions and attitudes toward COVID-19 vaccines.

Introduction

Vaccine confidence is an issue of public health concern, especially in the era of emerging and reemerging infectious diseases. It is defined by public perceptions of three components: vaccine safety, effectiveness, and importance.

Lack of trust and confidence in vaccines is an important determinant of vaccine hesitancy and influences vaccine uptake, affects national immunization targets, and increases outbreaks of vaccine-preventable diseases.

Hesitancy toward childhood and other vaccines has led to a resurgence of vaccine-preventable diseases such as measles and diarrhea. In some cases, it has affected polio eradication efforts, threatening years of progress in the global fight against infectious diseases.

The World Health Organization (WHO) identified reluctance or refusal to vaccinate as one of the top ten global health threats in 2019 that require monitoring and intervention by countries.
"""


# ─────────────────────────────────────────────
#  CACHED INDEX BUILDERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_preprocess(text, lowercase, rm_stopwords, do_stem, do_lemma):
    return preprocess(
        text,
        lowercase=lowercase,
        rm_stopwords=rm_stopwords,
        do_stem=do_stem,
        do_lemma=do_lemma,
    )


@st.cache_data(show_spinner=False)
def cached_build_biword(doc_items):
    return build_biword_index(dict(doc_items))


@st.cache_data(show_spinner=False)
def cached_build_positional(doc_items):
    return build_positional_index(dict(doc_items))


@st.cache_data(show_spinner=False)
def cached_build_inv_index(doc_items, lowercase, rm_stopwords, do_stem, do_lemma):
    return build_full_index(
        dict(doc_items),
        lowercase=lowercase,
        rm_stopwords=rm_stopwords,
        do_stem=do_stem,
        do_lemma=do_lemma,
    )


@st.cache_data(show_spinner=False)
def cached_build_vocab(doc_items, rm_stopwords, do_stem, do_lemma):
    return extract_vocabulary(
        dict(doc_items), rm_stopwords=rm_stopwords, do_stem=do_stem, do_lemma=do_lemma
    )


@st.cache_data(show_spinner=False)
def cached_build_kgram(vocab_tuple, k=3):
    return build_kgram_index(list(vocab_tuple), k=k)


@st.cache_data(show_spinner=False)
def cached_build_phonetic(vocab_tuple, method):
    return build_phonetic_index(list(vocab_tuple), method=method)


@st.cache_data(show_spinner=False)
def cached_tolerant_vocab(doc_items, rm_stopwords):
    return _get_vocabulary(dict(doc_items), rm_stopwords=rm_stopwords)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='padding:1.2rem 0 1rem;'>
        <div style='font-size:1.4rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.02em;'>🔍 IR System</div>
        <div style='font-size:0.75rem;color:#64748B;margin-top:2px;'>Assignment 1 — AIMLCZG537</div>
    </div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── File uploader ──
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
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.documents:
                content = f.read().decode("utf-8", errors="ignore")
                st.session_state.documents[f.name] = content

    if st.button("Load sample docs", use_container_width=True):
        st.session_state.documents.update(
            {
                "doc1.txt": "Facilitators and barriers to COVID-19 vaccine uptake among women in two regions of Ghana: A qualitative study. Although COVID-19 vaccines are available, evidence suggests that several factors hinder or facilitate their use. Several studies have found gender differences in COVID-19 vaccine uptake, with women less likely to vaccinate than men in many countries, including Ghana. These studies, however, have primarily been quantitative. This study used a qualitative approach to examine the facilitators and barriers to vaccine uptake among women in Ghana. Using a cross-sectional descriptive qualitative research design, 30 women in the Greater Accra and Ashanti regions of Ghana were conveniently sampled and interviewed using a semi-structured interview guide. Fifteen (15) interviews were conducted in each region. The data were transcribed verbatim and analysed thematically using QSR NVivo version 10 software. Among the key factors that facilitate COVID-19 vaccination are the desire to protect oneself and family against COVID-19, education about COVID-19 vaccines, seeing others receive the COVID-19 vaccine, and vaccine being cost-free. On the other hand, long queues at the vaccination centres, fear of side effects, misconceptions about the vaccines, and shortage of vaccines were the main barriers against COVID-19 vaccination. The study results show that individual, institutional, and vaccine-related factors facilitate or hinder COVID-19 vaccination among women. Addressing these factors need continuous comprehensive health education, and ensuring vaccine availability at vaccination sites will improve women’s uptake of the COVID-19 vaccines.",
                "doc2.txt": "The level and determinants of COVID-19 vaccine acceptance in Ghana. As part of the efforts to curb the COVID-19 pandemic, the government of Ghana has received several shipments of approved vaccines, and administration has begun in the country. Studies examining the determinants of COVID-19 vaccine acceptance in Ghana were mostly conducted before the vaccination exercise. Vaccine acceptance decisions however vary with time and hence, peoples’ decisions may have changed once vaccines became accessible. This study examines the level and determinants of COVID-19 vaccine acceptance among adult Ghanaians during the vaccination exercise. Methods. The study was a cross-sectional online survey involving Ghanaian adults (18 years and above) eligible to take the COVID-19 vaccine. The study was conducted from 18th May 2021 to 14th July 2021 and the questionnaire was answered by 362 respondents. Snowball sampling technique was utilized to obtain the respondents. Probit regression analysis was used to identify factors influencing COVID-19 vaccine acceptance.",
                "doc3.txt": text3,
                "doc4.txt": text4,
                "doc5.txt": "Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item.",
                "doc6.txt": "Information systems are used for retrieval of data. Retrieval information is key in modern search engines.",
                "doc7.txt": "Boolean retrieval model uses AND OR NOT operators to combine search terms.",
                "doc8.txt": "The inverted index maps each term to the list of documents containing that term.",
            }
        )
        st.success("Loaded 8 sample documents")

    if st.session_state.documents:
        if st.button("Clear all documents", use_container_width=True):
            for k in [
                "documents",
                "bw_index",
                "pos_index",
                "inv_index",
                "bst",
                "btree",
                "vocab",
            ]:
                st.session_state[k] = (
                    {} if k == "documents" else None if k != "vocab" else []
                )
            st.rerun()

    st.divider()

    # ── Query ──
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
#  HEADER STAT CARDS
# ─────────────────────────────────────────────
doc_count = len(st.session_state.documents)
total_words = sum(len(c.split()) for c in st.session_state.documents.values())

c1, c2, c3, c4 = st.columns(4)
for col, title, value in [
    (c1, "Documents", doc_count),
    (c2, "Total Words", f"{total_words:,}"),
    (c3, "Query", st.session_state.query or "—"),
    (c4, "Status", "Ready ✅" if doc_count else "No docs ⚠"),
]:
    col.markdown(
        f"""<div class='ir-card'>
        <div class='ir-card-title'>{title}</div>
        <div class='ir-card-value' style='font-size:{"1.8rem" if isinstance(value,int) else "1.1rem"};
             {"" if isinstance(value,int) else "padding-top:0.3rem;"}'>
            {value}
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

# ─────────────────────────────────────────────
#  TAB 1 — UPLOAD & PREVIEW
# ─────────────────────────────────────────────
with tab1:
    st.markdown("### Document collection")
    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>No documents loaded. Upload files or click <b>Load sample docs</b>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='ir-success'>✔ &nbsp;{doc_count} document{'s' if doc_count!=1 else ''} loaded &nbsp;·&nbsp; {total_words:,} total words</div>",
            unsafe_allow_html=True,
        )
        rows = [
            {
                "File": name,
                "Words": len(content.split()),
                "Characters": len(content),
                "Preview": content[:120].replace("\n", " ")
                + ("…" if len(content) > 120 else ""),
            }
            for name, content in st.session_state.documents.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### View full document")
        selected = st.selectbox(
            "Select document", list(st.session_state.documents.keys()), key="tab1_doc"
        )
        with st.expander("Full content", expanded=True):
            st.text(st.session_state.documents[selected])

# ─────────────────────────────────────────────
#  TAB 2 — PREPROCESSING
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### Text preprocessing pipeline")
    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>Upload documents first.</div>", unsafe_allow_html=True
        )
    else:
        doc_choice = st.selectbox(
            "Document to preprocess",
            list(st.session_state.documents.keys()),
            key="prep_doc",
        )
        raw_text = st.session_state.documents[doc_choice]

        # ── Run full pipeline ──
        result = cached_preprocess(raw_text, opt_lower, opt_stop, opt_stem, opt_lemma)

        st.markdown("#### Pipeline stages")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**① Raw text**")
            st.text_area("raw", raw_text, height=100, label_visibility="collapsed")

            st.markdown(
                f"**② After tokenization** &nbsp;<span class='badge badge-blue'>{len(result['raw_tokens'])} tokens</span>",
                unsafe_allow_html=True,
            )
            st.code(result["raw_tokens"][:30])

        with col2:
            st.markdown(
                f"**③ After lowercase** &nbsp;<span class='badge badge-green'>{'✔' if opt_lower else '—'}</span>",
                unsafe_allow_html=True,
            )
            st.code(result["lower_tokens"][:30])

            removed = result["removed_sw"]
            st.markdown(
                f"**④ After stop word removal** &nbsp;<span class='badge badge-amber'>{len(removed)} removed</span>",
                unsafe_allow_html=True,
            )
            st.code(result["clean_tokens"][:30])
            if removed:
                with st.expander(f"Removed stop words ({len(removed)})"):
                    st.code(removed)

            st.markdown(
                f"**⑤ Final tokens** &nbsp;<span class='badge badge-blue'>{len(result['final_tokens'])} tokens</span>",
                unsafe_allow_html=True,
            )
            st.code(result["final_tokens"][:30])

        # ── Stem / Lemma maps ──
        if result["stem_map"]:
            st.markdown("#### Stemming changes")
            st.dataframe(
                pd.DataFrame(
                    result["stem_map"].items(), columns=["Original", "Stemmed"]
                ),
                use_container_width=True,
                hide_index=True,
            )

        if result["lemma_map"]:
            st.markdown("#### Lemmatization changes")
            st.dataframe(
                pd.DataFrame(
                    result["lemma_map"].items(), columns=["Original", "Lemma"]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Inverted index ──
        st.markdown("#### Inverted index")
        with st.spinner("Building index..."):
            inv_index = cached_build_inv_index(
                tuple(st.session_state.documents.items()),
                opt_lower,
                opt_stop,
                opt_stem,
                opt_lemma,
            )
        st.session_state.inv_index = inv_index

        stats = index_statistics(inv_index, st.session_state.documents)
        s1, s2, s3 = st.columns(3)
        s1.metric("Vocabulary size", stats["vocab_size"])
        s2.metric("Total postings", stats["total_postings"])
        s3.metric("Avg doc frequency", stats["avg_df"])

        with st.expander("Full index table"):
            st.dataframe(
                full_index_df(inv_index), use_container_width=True, hide_index=True
            )

        st.divider()

        # ── Boolean retrieval ──
        st.markdown("#### Boolean retrieval")
        bool_query = st.text_input(
            "Boolean query (AND / OR / NOT)",
            placeholder="e.g.  information AND retrieval",
            key="bool_q",
        )
        if st.button("Run boolean query", key="run_bool") and bool_query:
            all_docs = set(st.session_state.documents.keys())
            bool_result = parse_boolena_query(bool_query, inv_index, all_docs)
            for step in bool_result["steps"]:
                st.markdown(f"<code>{step}</code>", unsafe_allow_html=True)
            if bool_result["results"]:
                st.markdown(
                    f"<div class='ir-success'>✔ Matched: {sorted(bool_result['results'])}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='ir-warn'>No documents matched.</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Stem vs Lemma comparison ──
        st.markdown("#### Stemming vs Lemmatization comparison")
        comp_input = st.text_input(
            "Queries to compare (comma separated)",
            value="information retrieval, search engine, word processing",
            key="comp_q",
        )
        if st.button("Run comparison", key="run_comp"):
            queries = [q.strip() for q in comp_input.split(",") if q.strip()]
            with st.spinner("Comparing..."):
                comp_df = compare_stem_vs_lemma(st.session_state.documents, queries)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            winner = (
                comp_df["Better"].value_counts().idxmax() if len(comp_df) else "Tie"
            )
            st.markdown(
                f"<div class='ir-success'>✔ &nbsp;<b>{winner}</b> performed better overall.</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
#  TAB 3 — PHRASE QUERY
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Phrase query — biword vs positional index")
    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>Upload documents first.</div>", unsafe_allow_html=True
        )
    else:
        doc_items = tuple(st.session_state.documents.items())

        # ── Build indexes ──
        with st.spinner("Building phrase indexes..."):
            bw_idx = cached_build_biword(doc_items)
            pos_idx = cached_build_positional(doc_items)

        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Biword index size", f"{len(bw_idx)} biwords")
        col_stat2.metric("Positional index size", f"{len(pos_idx)} terms")

        st.divider()

        # ── Index explorer ──
        with st.expander("🔎 Explore biword index"):
            st.dataframe(
                biwords_index_to_dataframe(bw_idx),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🔎 Explore positional index"):
            st.dataframe(
                positional_index_to_dataframe(pos_idx),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Query ──
        st.markdown("#### Run a phrase query")
        phrase = st.text_input(
            "Enter phrase query",
            value=st.session_state.query or "information retrieval",
            key="phrase_q",
            placeholder="e.g. information retrieval",
        )

        if st.button("Search phrase", key="run_phrase") or phrase:
            if phrase.strip():
                bw_result = query_biword(phrase, bw_idx)
                pos_result = query_postional(phrase, pos_idx)

                left, right = st.columns(2)

                # ── Biword results ──
                with left:
                    st.markdown("##### 📘 Biword index")
                    st.markdown(
                        f"<span class='badge badge-blue'>Biwords extracted: {len(bw_result['biwords'])}</span> &nbsp;"
                        f"<span class='badge badge-green'>Matched: {len(bw_result['results'])}</span> &nbsp;"
                        f"<span class='badge badge-amber'>{bw_result['elapsed_ms']} ms</span>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Processing steps"):
                        for step in bw_result["steps"]:
                            st.markdown(f"`{step}`")
                    if bw_result["results"]:
                        st.markdown(
                            "<div class='ir-success'>✔ Matched documents:<br>"
                            + "<br>".join(
                                f"&nbsp;&nbsp;• {d}"
                                for d in sorted(bw_result["results"])
                            )
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='ir-warn'>No documents matched.</div>",
                            unsafe_allow_html=True,
                        )

                # ── Positional results ──
                with right:
                    st.markdown("##### 📗 Positional index")
                    st.markdown(
                        f"<span class='badge badge-blue'>Candidates: {len(pos_result['candidates'])}</span> &nbsp;"
                        f"<span class='badge badge-green'>Matched: {len(pos_result['results'])}</span> &nbsp;"
                        f"<span class='badge badge-red'>False +ve eliminated: {len(pos_result['false_positives'])}</span> &nbsp;"
                        f"<span class='badge badge-amber'>{pos_result['elapsed_ms']} ms</span>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Processing steps"):
                        for step in pos_result["steps"]:
                            st.markdown(f"`{step}`")
                    if pos_result["results"]:
                        st.markdown(
                            "<div class='ir-success'>✔ Matched documents:<br>"
                            + "<br>".join(
                                f"&nbsp;&nbsp;• {d}"
                                for d in sorted(pos_result["results"])
                            )
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='ir-warn'>No documents matched.</div>",
                            unsafe_allow_html=True,
                        )

                # ── Comparison table ──
                st.markdown("#### Side-by-side comparison")
                comp_df = compare_indexes(
                    phrase, st.session_state.documents, bw_idx, pos_idx
                )
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── False positive demo ──
        st.markdown("#### False positive demonstration")
        st.markdown(
            "<div class='ir-info'>Shows a case where biword returns a wrong document that positional correctly rejects.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Run false positive demo", key="fp_demo"):
            demo = false_positive_demo()
            st.markdown(f"**Query:** `{demo['query']}`")

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Demo documents**")
                for name, text in demo["documents"].items():
                    st.markdown(f"`{name}`: {text}")

            with d2:
                st.markdown("**Results**")
                st.markdown(f"Biword matched: `{sorted(demo['bw_result']['results'])}`")
                st.markdown(
                    f"Positional matched: `{sorted(demo['pos_result']['results'])}`"
                )
                if demo["pos_result"]["false_positives"]:
                    st.markdown(
                        f"False positives eliminated: `{sorted(demo['pos_result']['false_positives'])}`"
                    )

            st.markdown(
                f"<div class='ir-warn'>⚠ {demo['explanation']}</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
#  TAB 4 — DICTIONARY SEARCH
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Dictionary search — BST vs B-Tree")
    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>Upload documents first.</div>", unsafe_allow_html=True
        )
    else:
        doc_items = tuple(st.session_state.documents.items())

        # ── Build trees ──
        with st.spinner("Building vocabulary and trees..."):
            vocab = cached_build_vocab(doc_items, opt_stop, opt_stem, opt_lemma)
            bst = build_bst_from_vocab(vocab)
            btree = build_btree_from_vocab(vocab)

        v1, v2, v3 = st.columns(3)
        v1.metric("Vocabulary size", len(vocab))
        v2.metric("BST height", bst.height())
        v3.metric("BST balanced", str(bst.is_balanced()))

        with st.expander("Tree structural comparison"):
            st.dataframe(
                tree_stats(bst, btree), use_container_width=True, hide_index=True
            )

        st.divider()

        # ── Benchmark ──
        st.markdown("#### Search benchmark")
        st.markdown(
            "<div class='ir-info'>Enter terms to search. At least one should be a real vocab term and one missing to show both cases.</div>",
            unsafe_allow_html=True,
        )

        default_queries = ", ".join(vocab[:3]) + ", missing_term_xyz"
        bench_input = st.text_input(
            "Query terms (comma separated)", value=default_queries, key="bench_q"
        )
        repeats = st.slider(
            "Repetitions per query (for stable timing)", 50, 500, 100, step=50
        )

        if st.button("Run benchmark", key="run_bench"):
            queries = [q.strip() for q in bench_input.split(",") if q.strip()]
            with st.spinner(f"Running {len(queries)} queries × {repeats} repeats..."):
                bench_df = benchmark(queries, bst, btree, n_repeats=repeats)
            st.dataframe(bench_df, use_container_width=True, hide_index=True)

            faster_counts = bench_df["Faster"].value_counts()
            winner = faster_counts.idxmax() if len(faster_counts) else "Tie"
            st.markdown(
                f"<div class='ir-success'>✔ <b>{winner}</b> was faster in most queries.</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── B-Tree extras ──
        st.markdown("#### B-Tree range & prefix search")
        rc1, rc2 = st.columns(2)
        with rc1:
            range_lo = st.text_input("Range from", value="a", key="range_lo")
            range_hi = st.text_input("Range to", value="c", key="range_hi")
            if st.button("Range query", key="run_range"):
                results = btree.range_query(range_lo, range_hi)
                st.markdown(
                    f"<div class='ir-success'>Found {len(results)} terms: {results}</div>",
                    unsafe_allow_html=True,
                )
        with rc2:
            prefix = st.text_input("Prefix search", value="inf", key="prefix_q")
            if st.button("Prefix query", key="run_prefix"):
                results = btree.prefix_search(prefix)
                st.markdown(
                    f"<div class='ir-success'>Found {len(results)} terms: {results}</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────
#  TAB 5 — TOLERANT RETRIEVAL
# ─────────────────────────────────────────────
with tab5:
    st.markdown("### Tolerant retrieval")
    if not st.session_state.documents:
        st.markdown(
            "<div class='ir-warn'>Upload documents first.</div>", unsafe_allow_html=True
        )
    else:
        doc_items = tuple(st.session_state.documents.items())

        # ── Build all tolerant indexes ──
        with st.spinner("Building tolerant indexes..."):
            t_vocab = cached_tolerant_vocab(doc_items, opt_stop)
            kg_index = cached_build_kgram(tuple(t_vocab), k=3)
            ph_soundex = cached_build_phonetic(tuple(t_vocab), "soundex")
            ph_meta = cached_build_phonetic(tuple(t_vocab), "metaphone")
            t_inv_idx = cached_build_inv_index(doc_items, True, opt_stop, False, False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Vocabulary size", len(t_vocab))
        m2.metric("K-gram index size", f"{len(kg_index)} k-grams")
        m3.metric("Phonetic groups", len(ph_soundex))

        st.markdown(
            "<div class='ir-info'>Enter a query below — use <code>*</code> for wildcard, or type a misspelled word to see correction.</div>",
            unsafe_allow_html=True,
        )

        t_subtab1, t_subtab2, t_subtab3, t_subtab4 = st.tabs(
            [
                "🔤 Wildcard",
                "✏️ Spelling Correction",
                "🔊 Phonetic",
                "🔗 Combined Search",
            ]
        )

        # ── WILDCARD ──
        with t_subtab1:
            st.markdown("#### Wildcard query via k-gram index")
            with st.expander("Browse k-gram index"):
                st.dataframe(
                    kgram_index_to_dataframe(kg_index, top_n=30),
                    use_container_width=True,
                    hide_index=True,
                )

            wc_input = st.text_input(
                "Wildcard pattern (* as wildcard)",
                value="inf*",
                key="wc_q",
                placeholder="e.g. inf*, *tion, s*em",
            )
            if st.button("Resolve wildcard", key="run_wc"):
                if "*" not in wc_input:
                    st.markdown(
                        "<div class='ir-warn'>Add * to your pattern e.g. inf*</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    wc_result = query_wildcard(
                        wc_input.strip().lower(), kg_index, t_vocab
                    )
                    with st.expander("Processing steps"):
                        for step in wc_result["steps"]:
                            st.markdown(f"`{step}`")

                    if wc_result["results"]:
                        st.markdown(
                            f"<div class='ir-success'>✔ Matched <b>{len(wc_result['results'])}</b> vocabulary terms: {wc_result['results']}</div>",
                            unsafe_allow_html=True,
                        )
                        # retrieve docs for matched terms
                        docs = set()
                        for term in wc_result["results"]:
                            docs |= set(t_inv_idx.get(term, {}).keys())
                        if docs:
                            st.markdown(
                                f"**Documents containing matched terms:** {sorted(docs)}"
                            )
                    else:
                        st.markdown(
                            "<div class='ir-warn'>No vocabulary terms matched this pattern.</div>",
                            unsafe_allow_html=True,
                        )

        # ── SPELLING CORRECTION ──
        with t_subtab2:
            st.markdown("#### Spelling correction via edit distance")
            sp_input = st.text_input(
                "Misspelled query term",
                value="infomation",
                key="sp_q",
                placeholder="e.g. infomation, retreival",
            )
            max_dist = st.slider("Max edit distance", 1, 4, 2, key="sp_dist")

            if st.button("Correct spelling", key="run_sp"):
                sp_result = spelling_correction(
                    sp_input.strip().lower(), t_vocab, top_n=5, max_dist=max_dist
                )
                with st.expander("Processing steps"):
                    for step in sp_result["steps"]:
                        st.markdown(f"`{step}`")

                if sp_result["suggestions"]:
                    st.markdown("**Correction suggestions:**")
                    st.dataframe(
                        spelling_correction_table(sp_result["suggestions"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    best = sp_result["best_match"]
                    st.markdown(
                        f"<div class='ir-success'>✔ Best match: <b>'{best}'</b></div>",
                        unsafe_allow_html=True,
                    )
                    docs = set(t_inv_idx.get(best, {}).keys())
                    if docs:
                        st.markdown(f"**Documents for '{best}':** {sorted(docs)}")
                    else:
                        st.markdown(
                            f"<div class='ir-warn'>No documents found for '{best}'.</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<div class='ir-warn'>No corrections found within edit distance.</div>",
                        unsafe_allow_html=True,
                    )

        # ── PHONETIC ──
        with t_subtab3:
            st.markdown("#### Phonetic correction (Soundex + Metaphone)")
            ph_input = st.text_input(
                "Query term for phonetic matching",
                value="serch",
                key="ph_q",
                placeholder="e.g. serch, natchural",
            )
            ph_method = st.radio(
                "Method", ["soundex", "metaphone"], horizontal=True, key="ph_method"
            )

            if st.button("Find phonetic matches", key="run_ph"):
                ph_idx = ph_soundex if ph_method == "soundex" else ph_meta
                ph_result = phonetic_correction(
                    ph_input.strip().lower(), ph_idx, method=ph_method
                )
                for step in ph_result["steps"]:
                    st.markdown(f"`{step}`")

                if ph_result["matches"]:
                    st.markdown(
                        f"<div class='ir-success'>✔ Phonetically similar terms: {ph_result['matches']}</div>",
                        unsafe_allow_html=True,
                    )
                    docs = set()
                    for term in ph_result["matches"]:
                        docs |= set(t_inv_idx.get(term, {}).keys())
                    if docs:
                        st.markdown(f"**Documents:** {sorted(docs)}")
                else:
                    st.markdown(
                        "<div class='ir-warn'>No phonetically similar terms found.</div>",
                        unsafe_allow_html=True,
                    )

            st.divider()
            with st.expander("Browse phonetic index groups (2+ terms)"):
                st.dataframe(
                    phonetic_index_to_dataframe(
                        ph_soundex if ph_method == "soundex" else ph_meta
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── COMBINED ──
        with t_subtab4:
            st.markdown("#### Combined tolerant search pipeline")
            st.markdown(
                "<div class='ir-info'>Handles wildcards, misspellings, and phonetic variations automatically.</div>",
                unsafe_allow_html=True,
            )

            comb_input = st.text_input(
                "Query (exact / misspelled / wildcard)",
                value="infomation",
                key="comb_q",
                placeholder="e.g. infomation, inf*, serch",
            )
            if st.button("Run tolerant search", key="run_comb"):
                with st.spinner("Running tolerant search..."):
                    comb_result = tolerant_search(
                        comb_input.strip(),
                        st.session_state.documents,
                        t_inv_idx,
                        kg_index,
                        ph_soundex,
                        ph_meta,
                        t_vocab,
                    )

                # show path taken
                if comb_result["is_wildcard"]:
                    st.markdown(
                        "<span class='badge badge-blue'>Wildcard path</span>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Wildcard steps"):
                        for s in comb_result["wildcard_result"]["steps"]:
                            st.markdown(f"`{s}`")
                elif comb_result["exact_match"]:
                    st.markdown(
                        "<span class='badge badge-green'>Exact match</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span class='badge badge-amber'>Tolerant correction applied</span>",
                        unsafe_allow_html=True,
                    )
                    col_sp, col_ph = st.columns(2)
                    with col_sp:
                        st.markdown("**Spelling correction**")
                        if comb_result["spell_result"]:
                            st.dataframe(
                                spelling_correction_table(
                                    comb_result["spell_result"]["suggestions"]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                    with col_ph:
                        st.markdown("**Phonetic matches (Soundex)**")
                        if comb_result["phonetic_soundex"]:
                            ph_matches = comb_result["phonetic_soundex"]["matches"]
                            st.markdown(
                                f"`Code: {comb_result['phonetic_soundex']['code']}`"
                            )
                            st.markdown(
                                f"Matches: {ph_matches if ph_matches else 'none'}"
                            )

                st.divider()
                st.markdown(f"**Resolved terms:** `{comb_result['final_terms']}`")
                if comb_result["retrieved_docs"]:
                    st.markdown(
                        f"<div class='ir-success'>✔ Retrieved <b>{len(comb_result['retrieved_docs'])}</b> document(s): "
                        + ", ".join(
                            f"<code>{d}</code>"
                            for d in sorted(comb_result["retrieved_docs"])
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='ir-warn'>No documents retrieved.</div>",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────
#  TAB 6 — INFERENCE
# ─────────────────────────────────────────────
with tab6:
    st.markdown("### Experimental results & inference")
    st.markdown(
        "<div class='ir-info'>Fill in your conclusions after running experiments in each tab.</div>",
        unsafe_allow_html=True,
    )

    questions = [
        (
            "Task B",
            "blue",
            "Which preprocessing technique improved retrieval quality the most?",
        ),
        (
            "Task B",
            "blue",
            "Was stemming or lemmatization better for your dataset? Why?",
        ),
        (
            "Task C",
            "green",
            "Which phrase query index was more accurate — biword or positional?",
        ),
        (
            "Task D",
            "amber",
            "Which tree structure was faster? Explain the time complexity difference.",
        ),
        (
            "Task E",
            "blue",
            "How effective was tolerant retrieval for misspelled and wildcard queries?",
        ),
        ("General", "amber", "What are the limitations of this system?"),
        ("General", "amber", "How can this system be improved in future?"),
    ]

    default_answers = {
        "Which preprocessing technique improved retrieval quality the most?": "Stop Word Removal significantly improved retrieval quality by eliminating common non-informative terms from the index. This reduced index noise and improved the effectiveness of retrieval operations.",
        "Was stemming or lemmatization better for your dataset? Why?": "Lemmatization proved superior for the selected dataset. It preserves valid dictionary forms of words and maintains semantic meaning more effectively than stemming.",
        "Which phrase query index was more accurate — biword or positional?": "The Positional Index was more accurate than the Biword Index because it stores exact term positions, enabling precise phrase matching and reducing false positives.",
        "Which tree structure was faster? Explain the time complexity difference.": "For small in-memory datasets, BST may occasionally perform faster due to simpler structure. However, B-Tree provides more stable performance and better scalability because it remains balanced.",
        "How effective was tolerant retrieval for misspelled and wildcard queries?": "The tolerant retrieval model was effective for imperfect queries. K-Gram Indexing, Levenshtein Edit Distance, wildcard processing, and phonetic matching helped handle spelling mistakes and incomplete terms.",
        "What are the limitations of this system?": "The system relies mainly on in-memory indexing and basic term matching. It does not perform semantic understanding and may require significant memory for large document collections.",
        "How can this system be improved in future?": "Future improvements may include BSBI indexing, vector-based semantic retrieval, advanced ranking algorithms, champion lists, and improved phonetic algorithms such as Double Metaphone.",
    }

    if "inference_answers" not in st.session_state:
        st.session_state.inference_answers = default_answers.copy()

    for tag, color, question in questions:
        st.markdown(
            f"<span class='badge badge-{color}'>{tag}</span>",
            unsafe_allow_html=True,
        )

        st.session_state.inference_answers[question] = st.text_area(
            question,
            value=st.session_state.inference_answers.get(question, ""),
            key=f"inf_{tag}_{question[:25]}",
            height=100,
        )

        st.markdown("")

    if st.button("Save Answers"):
        st.session_state.saved_inference_answers = (
            st.session_state.inference_answers.copy()
        )
        st.success("Answers saved successfully!")

    if "saved_inference_answers" in st.session_state:
        st.markdown("---")
        st.markdown("## 💡 Compulsory Evaluation Criteria Answers")

        for i, (question, answer) in enumerate(
            st.session_state.saved_inference_answers.items(), start=1
        ):
            st.markdown(f"### {i}. {question}")
            st.markdown(f"**Answer:** {answer}")
