import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SHL catalog
@st.cache_data
def load_data():
    df = pd.read_csv("shl_catalog.csv")
    df.fillna("", inplace=True)
    return df

df = load_data()

# Check available columns (adjust if needed)
# st.write(df.columns)  # You can uncomment this line to debug columns

# Combine available relevant text columns
df["combined"] = df[["Assessment Name", "Description", "Test Type"]].agg(" ".join, axis=1)

# Vectorize the combined descriptions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined"])

# UI
st.title("SHL Assessment Recommendation Engine")
query = st.text_area("Enter job description or query here:")

if st.button("Recommend"):
    if query.strip() == "":
        st.warning("Please enter a query!")
    else:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, X).flatten()
        top_indices = scores.argsort()[-10:][::-1]

        st.subheader("Top Recommendations:")
        for idx in top_indices:
            row = df.iloc[idx]
            st.markdown(f"### [{row['Assessment Name']}]({row['URL']})")
            st.write(f"**Duration**: {row['Duration']} | **Test Type**: {row['Test Type']}")
            st.write(f"**Remote Testing**: {row['Remote Testing Support']} | **Adaptive/IRT**: {row['Adaptive/IRT Support']}")
            st.write("---")
