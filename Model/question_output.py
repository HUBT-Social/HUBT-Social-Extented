from typing import List
from dataclasses import dataclass
import json
import re
@dataclass
class Answer:
    content: str
    def __init__(self, content: str):
        self.content = content

@dataclass
class Question:
    title: str
    answers: List[Answer]
    _correctAnswer: int
    def __init__(self, title: str,
                  answers: List[Answer],
                    correctAnswer: str):
        self.title = title
        self.answers = answers
        self.correctAnswer = correctAnswer
    
    
    @property
    def correctAnswer(self):
        return self._correctAnswer

    @correctAnswer.setter
    def correctAnswer(self, correctAnswer: str):
        TYPE = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

        match = re.search(r"đáp\s*án\s*[:.\-]?\s*([a-dA-D])", correctAnswer, re.IGNORECASE)
        if match:
            key = match.group(1).upper()
            if key in TYPE:
                self._correctAnswer = TYPE[key]
                return self._correctAnswer
        else:
            raise ValueError("Đáp án không hợp lệ. Chỉ chấp nhận A, B, C, D.")
        
    def json_convert(self):
        q_dict = {
            "title": self.title,
            "answers": [{"content": a.content} for a in self.answers],
            "correctAnswer": self._correctAnswer
        }
        return q_dict

    def __str__(self):
        q_dict = self.json_convert()
        return json.dumps(q_dict, ensure_ascii=False, indent=2)