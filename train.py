import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("spam.csv", sep='\t', names=['label','text'])

# Convert labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

# ---------------- KEYWORDS ----------------
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

# ---------------- FEATURES ----------------
df['length'] = df['text'].apply(len)

def count_keywords(text):
    text = str(text).lower()
    return sum(1 for word in spam_keywords if word in text)

df['keyword_count'] = df['text'].apply(count_keywords)

# ---------------- PREPARE DATA ----------------
X_text = df['text']
X_keywords = df['keyword_count']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_text)

X_final = hstack((X_vec, X_keywords.values.reshape(-1,1)))

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# ---------------- EVALUATION ----------------
print("Accuracy:", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ---------------- GRAPHS ----------------
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham")
plt.show()

plt.figure()
sns.histplot(df['length'], bins=50)
plt.title("Message Length Distribution")
plt.show()

plt.figure()
df['label'].value_counts().plot.pie(
    autopct='%1.1f%%',
    labels=['Ham', 'Spam']
)
plt.title("Spam Ratio")
plt.ylabel("")
plt.show()

plt.figure()
sns.boxplot(x='label', y='length', data=df)
plt.title("Length vs Spam")
plt.show()

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("✅ Model trained and saved successfully!")