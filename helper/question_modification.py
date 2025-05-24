import re
import pandas as pd
import joblib


CURRENT_RETURN_TYPE = ("", 0)


# Load mô hình và vectorizer
model = joblib.load("data/classifier_model.pkl")
vectorizer = joblib.load("data/vectorizer.pkl")

def classify_line_ml(line: str) -> str:
    global CURRENT_RETURN_TYPE
    vec = vectorizer.transform([line])
    CURRENT_RETURN_TYPE = (model.predict(vec)[0],0)
    return model.predict(vec)[0]

def classify_line(line):
    global CURRENT_RETURN_TYPE
    if re.match(r"^Câu\s*\d+", line, re.IGNORECASE):
        CURRENT_RETURN_TYPE = ("question", 0)
        return "question"
    elif re.match(r"^[a-dA-D]\.", line):
        if CURRENT_RETURN_TYPE[0] == "answer":
            CURRENT_RETURN_TYPE = ("answer", (CURRENT_RETURN_TYPE[1] + 1))
        else:
            CURRENT_RETURN_TYPE = ("answer", 0)
        return "answer"
    elif "Đáp án" in line or "đáp án" in line.lower():
        CURRENT_RETURN_TYPE = ("correct_answer", 0)
        return "correct_answer"
    else:
        if CURRENT_RETURN_TYPE[0] == "question":
            CURRENT_RETURN_TYPE = ("question", CURRENT_RETURN_TYPE[1] + 1)
            return "question"
        elif CURRENT_RETURN_TYPE == ("answer",3):
            CURRENT_RETURN_TYPE = ("correct_answer", 0)
            return "answer"
        
        return "other"

def question_modify(classified: list[tuple[str, str]]) -> list[dict]: 
    questions = []
    current_question = None
    collecting_question = False

    for line, label in classified:
        clean_line = line.replace("[QUESTION]", "").replace("[ANSWER]", "").replace("[CORRECT_ANSWER]", "").strip()

        if label == "question":
            # Nếu là bắt đầu câu hỏi mới
            if not collecting_question:
                # Lưu lại câu hỏi cũ (nếu có)
                if current_question:
                    if current_question["question"] and current_question["answers"] and current_question["correct"]: 
                        questions.append(current_question)
                current_question = {"question": [clean_line], "answers": [], "correct": None}
                collecting_question = True
            else:
                # Nếu tiếp tục là question => nối thêm vào nội dung
                current_question["question"].append(clean_line)

        elif label == "answer":
            if current_question:
                current_question["answers"].append(clean_line)
                collecting_question = False  # Chấm dứt nối câu hỏi nếu gặp answer

        elif label == "correct_answer":
            if current_question:
                current_question["correct"] = clean_line
                collecting_question = False

                # ✅ Sau khi có correct_answer → xử lý lại danh sách answers
                all_answers = current_question["answers"]
                all_questions = current_question["question"]
                if len(all_answers) > 4:

                    # Giữ lại 4 dòng gần nhất trước correct_answer
                    valid_answers = [ans for ans in all_answers[-4:]]
                    # lấy 4 cái gần nhất

                    # Những cái còn lại → trả về lại câu hỏi
                    extra_answers = [ans for ans in all_answers if ans not in valid_answers]

                    # Cập nhật lại danh sách answer
                    current_question["answers"] = [ans for ans in valid_answers]

                    # Chuyển các dòng dư thành nội dung câu hỏi
                    for extra in extra_answers:
                        current_question["question"].append(extra)
                if len(all_answers) < 2:
                    # Nếu chỉ có 1 đáp án, thì chuyển thành câu hỏi
                    valid_answers = [ans for ans in all_questions[-(4 - len(all_answers)):]] + all_answers
                    
                    remain_question = [ans for ans in all_questions if ans not in valid_answers]
                    
                    current_question["answers"] = [ans for ans in valid_answers]
                    current_question["question"] = remain_question

    # Thêm câu hỏi cuối cùng nếu chưa thêm
    if current_question["question"] and current_question["answers"] and current_question["correct"]: 
        questions.append(current_question)

    return questions

def dataframe_convert(questions: list[dict]) -> list[dict]:
    # Chuyển đổi thành DataFrame
    data = []
    for i, q in enumerate(questions, 1):
        # Nếu chưa có đáp án, xử lý bằng AI
        if not q["answers"] or not q["question"]:
            ai_classify = [(line, classify_line_ml(line)) for line in q["question"]]
            ai_classify_modify = question_modify(ai_classify)

            if len(ai_classify_modify) == 1:
                q["question"] = ai_classify_modify[0]["question"]
                q["answers"] = ai_classify_modify[0]["answers"]
            else:
                continue

        # Thêm bản gốc
        data.append({
            "Câu số": i,
            "Câu hỏi": q["question"],
            "Đáp án": [a for a in q["answers"]],
            "Đáp án đúng": q["correct"]
        })
        
        # Chuẩn hóa đáp án: bỏ a./b./c./d.
        answers_clean = [a[2:].strip() if a[:2].lower() in ["a.", "b.", "c.", "d."] else a for a in q["answers"]]

        data.append({
            "Câu số": i,
            "Câu hỏi": q["question"],
            "Đáp án": answers_clean,
            "Đáp án đúng": q["correct"]
        })

        # 📌 THÊM BẢN BIẾN THỂ: nếu câu hỏi bắt đầu bằng "Câu xxx:", tạo thêm bản sao có "câuxxx:"
        match = re.match(r"Câu\s*(\d+)", q["question"][0], re.IGNORECASE)
        if match:
            number = match.group(1)
            alt_question = re.sub(r"Câu\s*%s" % number, f"câu{number}", q["question"][0], flags=re.IGNORECASE)
            data.append({
                "Câu số": f"{i}_alt",
                "Câu hỏi": [alt_question],
                "Đáp án": answers_clean,
                "Đáp án đúng": q["correct"]
            })

    df = pd.DataFrame(data)

    # Ghi ra file Excel
    df.to_excel("output.xlsx", index=False)

    # Ghi ra file JSON
    df.to_json("output.json", force_ascii=False, orient="records", indent=2)

if __name__ == "__main__":
    # Ví dụ sử dụng
    classified_data = [
        ("[QUESTION] Câu 200:  Chương trình thực thi MPI với ô checkbox được đánh dấu như sau dùng để làm gì?", "question"),
        ("[ANSWER] Để các tiến trình chạy trên các cửa sổ độc lập với nhau", "answer"),
        ("[ANSWER] Để chạy bình thường trong cửa sổ", "answer"),  # ❌ gán nhầm
        ("[ANSWER] Để các tiến trình chạy trên các cửa sổ command prompt độc lập với nhau", "answer"),
        ("[ANSWER] Không câu nào đúng", "answer"),  # ❌ gán nhầm
        ("[CORRECT_ANSWER] Đáp án: a", "correct_answer"),
        ("[QUESTION] Câu 201:  Chương trình thực thi MPI với ô checkbox được đánh dấu như sau dùng để làm gì?", "question"),
        ("[ANSWER] Để các tiến trình chạy trên các cửa sổ độc lập với nhau", "answer"),
        ("[QUESTION] Để chạy bình thường trong cửa sổ", "question"),  # ❌ gán nhầm
        ("[ANSWER] Để các tiến trình chạy trên các cửa sổ command prompt độc lập với nhau", "answer"),
        ("[QUESTION] Không câu nào đúng", "question"),  # ❌ gán nhầm
        ("[CORRECT_ANSWER] Đáp án: a", "correct_answer"),
    ]   

    processed_questions = question_modify(classified_data)
    print(processed_questions)
