import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def learning_set():
    # Load dữ liệu huấn luyện
    df = pd.read_json("output.json")
    # Duyệt từng dòng và tạo tập dữ liệu huấn luyện
    training_data = []

    for row in df.to_dict(orient="records"):
        # Câu hỏi
        for q in row["Câu hỏi"]:
            training_data.append({
                "text": q,
                "label": "question"
            })

        # Đáp án A–D
        for anwser in row["Đáp án"]:
            training_data.append({
                "text": anwser,
                "label": "answer"
            })

        # Đáp án đúng
        if "Đáp án đúng" in row and isinstance(row["Đáp án đúng"], str):
            training_data.append({
                "text": row["Đáp án đúng"],
                "label": "correct_answer"
            })
        
    # Tạo DataFrame huấn luyện
    train_df = pd.DataFrame(training_data)

    # Huấn luyện mô hình
    X = train_df["text"]
    y = train_df["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    # Tạo thư mục nếu chưa có
    os.makedirs("data", exist_ok=True)

    # Lưu model và vectorizer
    joblib.dump(model, "data/classifier_model.pkl")
    joblib.dump(vectorizer, "data/vectorizer.pkl")

    print("✅ Huấn luyện và lưu mô hình thành công!")
def update_model(new_text, true_label):

    vectorizer = joblib.load("data/vectorizer.pkl")
    model = joblib.load("data/classifier_model.pkl")

    X_new = vectorizer.transform([new_text])
    y_new = [true_label]

    model.partial_fit(X_new, y_new, classes=["question", "answer", "correct_answer", "other"])

    # Ghi đè mô hình
    joblib.dump(model, "data/classifier_model.pkl")

    print("✅ Mô hình đã cập nhật học từ phản hồi mới!")