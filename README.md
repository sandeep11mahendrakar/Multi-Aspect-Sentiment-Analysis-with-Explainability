col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Overall Sentiment")
    st.markdown(f"### {pred.upper()}")
    st.metric("Confidence", f"{max(prob)*100:.2f}%")

with col2:
    st.subheader("Probability Distribution")

    prob_df = pd.DataFrame({
        "Sentiment": ["Negative", "Neutral", "Positive"],
        "Probability": prob
    })

    st.bar_chart(prob_df.set_index("Sentiment"))
