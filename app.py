import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# 🎨 Setup & Gaya
# =============================
st.set_page_config(page_title="Sentimen myPertamina — Simple", layout="wide")

st.markdown(
    """
    <style>
      .block-container{padding-top:2rem;padding-bottom:2rem;}
      [data-testid="stSidebar"] {background:#0f172a;color:#e5e7eb}
      [data-testid="stSidebar"] * {color:#e5e7eb}
      .kpi{padding:1rem;border-radius:1rem;background:#0ea5e9;color:white}
      .kpi h4{margin:0;font-size:.9rem;opacity:.9}
      .kpi p{margin:.25rem 0 0 0;font-size:1.4rem;font-weight:700}
      .note{padding:.6rem .8rem;border:1px dashed #94a3b8;border-radius:.75rem;background:#0b1220;color:#cbd5e1}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# 🧰 Utils
# =============================
TEXT_CANDIDATES = ["content","review","ulasan","text","comment","comments","body","title"]
LABEL_CANDIDATES = ["label","labels","sentiment","score","rating","stars"]

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_binary(y):
    y = pd.to_numeric(y, errors='coerce')
    if pd.isna(y):
        return np.nan
    if y in [1,2]:
        return 0
    if y in [4,5]:
        return 1
    return np.nan  # buang 3 / nilai lain

@st.cache_data(show_spinner=False)
def build_clean_df(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Bangun dataframe yang sudah di pra-pemrosesan sesuai skema biner."""
    work = df.dropna(subset=[text_col, label_col]).copy()
    work["_y"] = work[label_col].apply(to_binary)
    work = work.dropna(subset=["_y"]).copy()
    work["_y"] = work["_y"].astype(int)
    work["_x"] = work[text_col].astype(str).apply(clean_text)
    return work

# =============================
# 📂 Sidebar Menu
# =============================
st.sidebar.title("📂 Menu")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard", "Upload Data", "Pra-Pemrosesan", "Latih Model", "Prediksi"], index=0
)

# Shared state
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "text_col" not in st.session_state:
    st.session_state.text_col = None
if "label_col" not in st.session_state:
    st.session_state.label_col = None

# =============================
# 🏠 Dashboard
# =============================
if menu == "Dashboard":
    st.header("🏠 Dashboard Analisis Sentimen MyPertamina")
    st.caption("Ringkasan cepat dataset, distribusi label, panjang teks, serta sampel ulasan.")

    if st.session_state.df is None:
        st.info("Belum ada data. Silakan unggah di menu **Upload Data**.")
    else:
        df = st.session_state.df.copy()
        all_cols = df.columns.tolist()
        text_col = st.session_state.text_col or next((c for c in TEXT_CANDIDATES if c in all_cols), all_cols[0])
        label_col = st.session_state.label_col or next((c for c in LABEL_CANDIDATES if c in all_cols), (all_cols[1] if len(all_cols)>1 else all_cols[0]))

        k1,k2,k3,k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='kpi'><h4>Baris</h4><p>{len(df):,}</p></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'><h4>Kolom teks</h4><p>{text_col}</p></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi'><h4>Kolom label</h4><p>{label_col}</p></div>", unsafe_allow_html=True)
        with k4:
            nunique = df[label_col].nunique(dropna=True) if label_col in df else 0
            st.markdown(f"<div class='kpi'><h4>Unik Label</h4><p>{nunique}</p></div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Distribusi Label (Raw)")
            if label_col in df:
                vc = df[label_col].value_counts(dropna=False).reset_index()
                vc.columns = [label_col, 'jumlah']
                fig = px.bar(vc, x=label_col, y='jumlah', text='jumlah')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom label belum ada.")
        with colB:
            st.subheader("Panjang Teks (Raw)")
            if text_col in df:
                lens = df[text_col].astype(str).str.len()
                fig2 = px.histogram(pd.DataFrame({"len": lens}), x='len', nbins=40)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Kolom teks belum ada.")

        with st.expander("🔎 Sampel Ulasan (10 baris)"):
            sample_cols = [c for c in [text_col, label_col] if c in df.columns]
            st.dataframe(df[sample_cols].head(10), use_container_width=True)

# =============================
# 🗂️ Upload Data
# =============================
elif menu == "Upload Data":
    st.header("📥 Upload / Muat Dataset")
    st.caption("CSV dengan minimal kolom teks & label/rating. Skema biner: 1–2=negatif, 4–5=positif (3 dibuang).")

    colL, colR = st.columns([2,1])
    with colL:
        file = st.file_uploader("Unggah CSV", type=["csv"], help="Format umum: content, score")
    with colR:
        if st.button("Gunakan Data Alternatif: myPertamina.csv", use_container_width=True):
            try:
                st.session_state.df = pd.read_csv("myPertamina.csv")
                st.success("Contoh myPertamina.csv dimuat.")
            except Exception as e:
                st.error(f"Gagal memuat myPertamina.csv: {e}")

    if file is not None:
        try:
            st.session_state.df = pd.read_csv(file)
            st.success("Dataset berhasil dimuat dari upload.")
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        all_cols = df.columns.tolist()
        text_default = next((c for c in TEXT_CANDIDATES if c in all_cols), all_cols[0])
        label_default = next((c for c in LABEL_CANDIDATES if c in all_cols), (all_cols[1] if len(all_cols)>1 else all_cols[0]))

        st.session_state.text_col = st.selectbox("Kolom teks", options=all_cols, index=all_cols.index(text_default))
        st.session_state.label_col = st.selectbox("Kolom label/rating", options=all_cols, index=all_cols.index(label_default))

        with st.expander("🔎 Pratinjau data (5 baris)", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        k1,k2,k3 = st.columns(3)
        with k1:
            st.markdown(f"<div class='kpi'><h4>Baris</h4><p>{len(df):,}</p></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'><h4>Kolom teks</h4><p>{st.session_state.text_col}</p></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi'><h4>Kolom label</h4><p>{st.session_state.label_col}</p></div>", unsafe_allow_html=True)
    else:
        st.info("Unggah CSV atau klik tombol untuk memuat contoh.")

# =============================
# 🔄 Pra-Pemrosesan (Perbandingan)
# =============================
elif menu == "Pra-Pemrosesan":
    st.header("🔄 Perbandingan Data: Sebelum vs Sesudah Pra-Pemrosesan")
    if st.session_state.df is None:
        st.warning("Silakan unggah data di menu **Upload Data** terlebih dahulu.")
        st.stop()

    df_raw = st.session_state.df.copy()
    text_col = st.session_state.text_col
    label_col = st.session_state.label_col

    if not text_col or not label_col:
        st.error("Kolom teks atau label belum dipilih di menu **Upload Data**.")
        st.stop()

    # Bangun data hasil pra-pemrosesan
    clean_df = build_clean_df(df_raw, text_col, label_col)

    # Info ringkas
    raw_rows = len(df_raw)
    after_dropna = len(df_raw.dropna(subset=[text_col, label_col]))
    after_map = len(clean_df)
    dropped_total = raw_rows - after_map

    st.markdown(
        f"<div class='note'>Baris awal: <b>{raw_rows:,}</b> • Setelah drop NA: <b>{after_dropna:,}</b> • Setelah mapping & cleaning: <b>{after_map:,}</b> • Dibuang total: <b>{dropped_total:,}</b></div>",
        unsafe_allow_html=True,
    )

    n_show = st.slider("Tampilkan berapa baris sampel?", min_value=5, max_value=30, value=10, step=1)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Sebelum Pra-Pemrosesan (Raw)")
        cols = [c for c in [text_col, label_col] if c in df_raw.columns]
        st.dataframe(df_raw[cols].head(n_show), use_container_width=True)
    with col2:
        st.subheader("✅ Sesudah Pra-Pemrosesan (Clean)")
        st.dataframe(
            clean_df[["_x", "_y"]].head(n_show).rename(columns={"_x": "clean_text", "_y": "label_biner"}),
            use_container_width=True,
        )

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Distribusi Label (Raw vs Biner)")
        try:
            raw_counts = df_raw[label_col].value_counts(dropna=False).reset_index()
            raw_counts.columns = ["label_raw", "jumlah"]
            raw_counts["tipe"] = "raw"
            bin_counts = clean_df["_y"].value_counts(dropna=False).reset_index()
            bin_counts.columns = ["label_raw", "jumlah"]
            bin_counts["tipe"] = "biner"
            both = pd.concat([raw_counts, bin_counts], ignore_index=True)
            figc = px.bar(both, x="label_raw", y="jumlah", color="tipe", barmode="group", text="jumlah")
            st.plotly_chart(figc, use_container_width=True)
        except Exception:
            st.info("Tidak bisa membuat chart distribusi label.")
    with cB:
        st.subheader("Panjang Teks: Raw vs Clean")
        try:
            tmp = pd.DataFrame({
                "len_raw": df_raw[text_col].astype(str).str.len(),
                "len_clean": clean_df["_x"].astype(str).str.len()
            }).melt(var_name="tipe", value_name="panjang")
            figl = px.histogram(tmp, x="panjang", color="tipe", barmode="overlay", nbins=50)
            st.plotly_chart(figl, use_container_width=True)
        except Exception:
            st.info("Tidak bisa membuat chart panjang teks.")

    with st.expander("ℹ️ Penjelasan Tahapan Pra-Pemrosesan"):
        st.markdown(
            "- **Hapus data kosong** pada kolom teks dan label.\n"
            "- **Konversi label**: Rating 1–2 → `0` (Negatif), 4–5 → `1` (Positif), nilai 3/aneh dibuang.\n"
            "- **Cleaning teks**: lowercase → hapus non-huruf → normalisasi spasi.\n"
        )

# =============================
# 🧠 Latih Model (dengan visual split)
# =============================
elif menu == "Latih Model":
    st.header("🧠 Latih Model")
    if st.session_state.df is None:
        st.warning("Silakan muat data di menu **Upload Data** dulu.")
        st.stop()

    df = st.session_state.df.copy()
    text_col = st.session_state.text_col or df.select_dtypes(include=['object']).columns[0]
    label_col = st.session_state.label_col or (set(df.columns)-{text_col}).pop()

    # Siapkan target biner & teks bersih
    df = df.dropna(subset=[text_col, label_col]).copy()
    df["_y"] = df[label_col].apply(to_binary)
    dropped = int(df["_y"].isna().sum())
    df = df.dropna(subset=["_y"]).copy()
    df["_y"] = df["_y"].astype(int)
    df["_x"] = df[text_col].astype(str).apply(clean_text)

    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    with c2:
        random_state = st.number_input("random_state", value=42, step=1)
    with c3:
        st.caption(f"Skema biner: 1–2→0, 4–5→1. Dibuang: {dropped} baris (nilai 3/tidak valid).")

    stratify = df["_y"] if df["_y"].nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df["_x"], df["_y"], test_size=test_size, random_state=random_state, stratify=stratify
    )

    # =============================
    # 🎯 Visualisasi Pembagian Dataset
    # =============================
    st.subheader("📊 Visualisasi Pembagian Dataset (Train/Test)")
    train_n, test_n = len(X_train), len(X_test)
    total_n = train_n + test_n if (len(X_train) + len(X_test)) > 0 else 1
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"**Train**: {train_n:,} baris ({(train_n/total_n)*100:.1f}%)")
    with k2:
        st.markdown(f"**Test**: {test_n:,} baris ({(test_n/total_n)*100:.1f}%)")
    with k3:
        st.markdown(f"**Stratify**: {'Ya' if stratify is not None else 'Tidak'}")

    # Pie chart komposisi train/test
    pie_df = pd.DataFrame({"split": ["Train","Test"], "jumlah": [train_n, test_n]})
    pie_fig = px.pie(pie_df, names="split", values="jumlah", title="Proporsi Train vs Test")

    # Distribusi label per split
    tr_counts = y_train.value_counts().reset_index(); tr_counts.columns = ["label","jumlah"]; tr_counts["split"] = "Train"
    te_counts = y_test.value_counts().reset_index(); te_counts.columns = ["label","jumlah"]; te_counts["split"] = "Test"
    dist_df = pd.concat([tr_counts, te_counts], ignore_index=True)
    bar_fig = px.bar(dist_df, x="label", y="jumlah", color="split", barmode="group", text="jumlah", title="Distribusi Label per Split")

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(pie_fig, use_container_width=True)
    with cB:
        st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("🔎 Sampel Tiap Split (5 baris)"):
        colL, colR = st.columns(2)
        with colL:
            st.caption("Train")
            st.dataframe(pd.DataFrame({"text": X_train.head(5), "label": y_train.head(5)}), use_container_width=True)
        with colR:
            st.caption("Test")
            st.dataframe(pd.DataFrame({"text": X_test.head(5), "label": y_test.head(5)}), use_container_width=True)

    # =============================
    # 🚀 Training Model
    # =============================
    if st.button("🚀 Latih LinearSVC", type="primary"):
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
        clf = LinearSVC()
        with st.spinner("Melatih model..."):
            Xtr = vectorizer.fit_transform(X_train)
            Xte = vectorizer.transform(X_test)
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

        st.session_state.model = clf
        st.session_state.vectorizer = vectorizer

        st.success(f"Model selesai dilatih. Akurasi: {acc:.2%}")

        rep_df = pd.DataFrame(report).transpose()
        cols = [c for c in ["precision","recall","f1-score","support"] if c in rep_df.columns]
        st.subheader("📈 Evaluasi Model")
        st.dataframe(rep_df[cols].style.format({"precision":"{:.2f}","recall":"{:.2f}","f1-score":"{:.2f}","support":"{:.0f}"}), use_container_width=True)

        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        labels = np.unique(y_test)
        ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
        st.pyplot(fig)

        try:
            st.subheader("🔎 Kata/N-gram Paling Berpengaruh")
            feature_names = np.array(st.session_state.vectorizer.get_feature_names_out())
            coefs = st.session_state.model.coef_.ravel()
            top_pos = feature_names[np.argsort(coefs)[-15:]][::-1]
            top_neg = feature_names[np.argsort(coefs)[:15]]
            cpos, cneg = st.columns(2)
            with cpos:
                st.write("🌟 Positif")
                st.write(pd.DataFrame({"n-gram": top_pos}))
            with cneg:
                st.write("⚠️ Negatif")
                st.write(pd.DataFrame({"n-gram": top_neg}))
        except Exception:
            pass

        with st.expander("💾 Ekspor Model"):
            buf = io.BytesIO()
            joblib.dump({"vectorizer": vectorizer, "model": clf}, buf)
            st.download_button("Download joblib", data=buf.getvalue(), file_name="mypertamina_sentiment.joblib", mime="application/octet-stream")

# =============================
# 📝 Prediksi
# =============================
elif menu == "Prediksi":
    st.header("📝 Prediksi Sentimen")
    if st.session_state.model is None or st.session_state.vectorizer is None:
        st.info("Latih model dulu di menu **Latih Model**.")
        st.stop()

    txt = st.text_area("Masukkan teks ulasan:", height=120, placeholder="Contoh: Aplikasinya mudah dipakai dan responsif…")
    if st.button("Prediksi", use_container_width=True):
        if not txt.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            x = clean_text(txt)
            Xv = st.session_state.vectorizer.transform([x])
            pred = int(st.session_state.model.predict(Xv)[0])
            score = float(st.session_state.model.decision_function(Xv)[0])
            label = "🌟 Positif" if pred==1 else "⚠️ Negatif"
            st.success(f"Hasil: {label}")
            st.caption(f"Skor keputusan: {score:.3f} (semakin besar → semakin positif)")

