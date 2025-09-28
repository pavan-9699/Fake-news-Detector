import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st

def train_model():
    true_data = pd.read_csv("true.csv")
    true_data['label'] = 0
    fake_data = pd.read_csv("fake.csv")
    fake_data['label'] = 1
    data = pd.concat([true_data, fake_data], ignore_index=True)
    data = data[['text', 'label']].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

def load_html(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

    load_html("style.html")  # Load HTML + CSS

    st.markdown("<h1 class='main-title'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Paste a news article below to check if it's Real or Fake.</p>", unsafe_allow_html=True)

    if "news_input" not in st.session_state:
        st.session_state.news_input = ""

    try:
        true_data = pd.read_csv("true.csv")
        fake_data = pd.read_csv("fake.csv")
        total = len(true_data) + len(fake_data)
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Real News Samples", len(true_data))
        col2.metric("🔴 Fake News Samples", len(fake_data))
        col3.metric("📄 Total Records", total)
        if st.button("🎲 Try Sample News"):
            all_data = pd.concat([true_data, fake_data], ignore_index=True)
            sample = all_data.sample(n=1).iloc[0]
            st.session_state.news_input = sample['text']
            st.info("📝 Sample News Loaded!")
    except:
        st.warning("Dataset preview not available (CSV files missing or unreadable).")

    input_text = st.text_area("Enter News Article", value=st.session_state.news_input, height=250, placeholder="Paste the news text here...")
    if st.button("Predict"):
        if not input_text.strip():
            st.warning("⚠️ Please enter some news text before clicking Predict.")
        else:
            st.session_state.news_input = input_text
            model, tfidf = load_model()
            vector = tfidf.transform([input_text])
            prediction = model.predict(vector)[0]
            if prediction == 0:
                st.success("🟢 Prediction: This news is **Real**.")
            else:
                st.error("🔴 Prediction: This news is **Fake**.")

    st.sidebar.header("📌 About")
    st.sidebar.info(
        """
        This app uses a Logistic Regression model trained on a dataset of real and fake news.
        Enter any news article in the text box to see whether it is classified as Real or Fake.
        Built with Python, Scikit-learn & Streamlit.
        """
    )

    st.markdown("<footer>Made with ❤️ by Pavan | Logistic Regression + Streamlit</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    train_model()
    main()
