Saya akan menambahkan visualisasi pembagian dataset pada bagian menu **Latih Model**. Visualisasi ini akan menggunakan diagram batang untuk menunjukkan proporsi data latih dan data uji.

```python
# ... (kode sebelumnya tetap sama hingga bagian Latih Model)

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

    # Visualisasi pembagian dataset
    st.subheader("📊 Visualisasi Pembagian Dataset")
    split_df = pd.DataFrame({
        'Set': ['Train'] * len(y_train) + ['Test'] * len(y_test),
        'Label': list(y_train) + list(y_test)
    })
    fig_split = px.histogram(split_df, x='Label', color='Set', barmode='group', text_auto=True)
    st.plotly_chart(fig_split, use_container_width=True)

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

# ... (lanjutkan kode setelahnya tanpa perubahan)
```
