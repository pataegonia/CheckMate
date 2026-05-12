"""서버에 저장된 문항 정보를 흉내내는 dict.

type:
  - multiple_choice: 객관식. 표시/동그라미/체크를 정답 번호와 비교한다.
  - short_answer: 서답형. 풀이가 섞여 있어도 최종 답을 찾아 정답과 비교한다.
  - descriptive: 서술형. 풀이 과정과 채점 기준까지 확인한다.
"""

QUESTION_SPECS = {
    814: {
        "answer": "1",
        "type": "multiple_choice",
        "rubric": "선택한 객관식 번호가 정답과 일치하면 1점.",
    },
    815: {
        "answer": "2",
        "type": "multiple_choice",
        "rubric": "선택한 객관식 번호가 정답과 일치하면 1점.",
    },
    816: {
        "answer": "5",
        "type": "multiple_choice",
        "rubric": "선택한 객관식 번호가 정답과 일치하면 1점.",
    },
    817: {
        "answer": "4",
        "type": "multiple_choice",
        "rubric": "선택한 객관식 번호가 정답과 일치하면 1점.",
    },
    818: {
        "answer": "x<-3",
        "type": "short_answer",
        "rubric": "풀이 과정 중 최종 답을 찾아 정답과 동치이면 1점.",
    },
    819: {
        "answer": "-4",
        "type": "short_answer",
        "rubric": "풀이 과정 중 최종 답을 찾아 정답과 동치이면 1점.",
    },
    820: {
        "answer": "x<-2",
        "type": "short_answer",
        "rubric": "풀이 과정 중 최종 답을 찾아 정답과 동치이면 1점.",
    },
}

ANSWERS = {num: spec["answer"] for num, spec in QUESTION_SPECS.items()}


def get_question_spec(problem_num: int) -> dict:
    return QUESTION_SPECS.get(
        problem_num,
        {
            "answer": ANSWERS.get(problem_num, ""),
            "type": "short_answer",
            "rubric": "",
        },
    )
