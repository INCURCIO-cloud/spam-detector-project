import streamlit as st
import pickle
from scipy.sparse import hstack

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ---------------- KEYWORDS (SAME AS TRAIN) ----------------
spam_keywords = [
    "win","won","winner","prize","lottery","jackpot",
    "cash","money","reward","bonus","income","earn",
    "claim","refund","payment","gift","free money",
    "urgent","immediately","important","final warning",
    "last chance","expires today","limited time","hurry",
    "asap","verify immediately","account suspended",
    "click","click here","visit","download","access",
    "login","sign in","http","https","bit.ly",
    "free","offer","discount","deal","promo",
    "congratulations","selected","exclusive","buy now",
    "bank","account","password","pin","otp",
    "verify","security","update","credit card",
    "debit card","blocked","suspended",
    "call now","contact","whatsapp","reply",
    "send details","personal details","phone number"
]

# ---------------- FUNCTION ----------------
def count_keywords(text):
    text = text.lower()
    return sum(1 for word in spam_keywords if word in text)

# ---------------- UI ----------------
st.title("📧 Smart Spam Email Detector")

email = st.text_area("Enter your email/message here:")

if st.button("Check Spam"):
    if email.strip() == "":
        st.warning("Please enter some text!")
    else:
        # TF-IDF
        vec = vectorizer.transform([email])

        # Keyword feature
        keyword_feature = [[count_keywords(email)]]

        # Combine
        final_input = hstack((vec, keyword_feature))

        # Prediction
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1]

        # Output
        if prediction == 1:
            st.error(f"🚨 Spam Detected!\nConfidence: {prob:.2f}")
        else:
            st.success(f"✅ Not Spam\nConfidence: {1 - prob:.2f}")