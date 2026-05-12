# CheckMate AI 로직 작업 계획

## 우리 역할

AI 파트는 OCR을 포함한 전체 채점 파이프라인을 담당한다.

1. 이미지/PDF에서 문제 번호와 문항 영역을 찾는다.
2. 객관식 표시와 단답형 손글씨 답안을 인식한다.
3. 정답/채점 기준과 비교해 1차 채점 결과를 만든다.
4. 불확실하거나 서술형 채점이 필요한 문항은 Bedrock/Claude 검토 대상으로 보낸다.
5. 채점 결과를 백엔드가 저장할 수 있는 JSON 형태로 반환한다.
6. 재채점, 챗봇, 유사문제 생성, 취약 유형 분석으로 확장한다.

## 빠른 MVP

현재 Streamlit 데모를 유지하면서 아래 흐름을 먼저 완성한다.

`PDF/이미지 업로드 -> OCR/문항 분할 -> 답안 후보 추출 -> rule 기반 채점 -> AI 검토 필요 여부 표시 -> 결과 JSON 생성`

이 단계에서는 AWS를 붙이지 않아도 발표 시연이 가능하다. Bedrock 연동 전에는 `MockAIGrader`로 Claude 응답처럼 보이는 결과를 만든다.

## 하이브리드 채점 전략

모든 문항을 Claude에 보내지 않고, 빠르고 안정적인 로컬 로직과 생성형 AI를 나눠 쓴다.

- 객관식은 표시/동그라미/체크를 우선 감지하고, 실패하면 Claude가 crop 이미지를 보고 선택지를 판단한다.
- 서답형은 OCR 후보와 crop 이미지를 함께 사용하여 여러 풀이 과정 중 최종 답을 추출한다.
- 서술형은 crop 이미지, 정답, rubric을 Claude에 전달하여 풀이 과정과 감점 사유를 판단한다.
- 재채점 요청은 원본 이미지, 이전 결과, 채점 기준을 함께 넣어 AI가 다시 판단

성능 전략:

- 같은 업로드 페이지의 OCR/문항 분할 결과는 Streamlit cache에 저장한다.
- 기본 시연은 `빠른 모드: OCR/rule`로 수행한다.
- 발표에서 AI를 보여줄 때는 `하이브리드: 불확실/서술형만 Claude` 또는 `서답형+서술형 Claude`를 사용한다.
- `전체 Claude`는 비교 시연 또는 디버깅 때만 사용한다.

## 백엔드에 제안할 결과 JSON

```json
{
  "submissionId": "local-demo-submission",
  "assignmentId": "local-demo-assignment",
  "studentId": "local-demo-student",
  "totalScore": 5.0,
  "maxScore": 7.0,
  "questions": [
    {
      "questionId": "814",
      "problemNum": 814,
      "type": "multiple_choice",
      "extractedAnswer": "1",
      "isCorrect": true,
      "score": 1.0,
      "maxScore": 1.0,
      "reason": "OCR로 추출한 답안이 저장된 정답과 일치합니다.",
      "confidence": 0.91,
      "source": "rule",
      "needsReview": false,
      "deduction": ""
    }
  ]
}
```

## Bedrock/Claude 프롬프트 원칙

- 응답은 JSON만 받는다.
- 점수, 정오답, 감점 사유, 판단 근거, 신뢰도, 재검토 필요 여부를 반드시 포함한다.
- 교사가 입력한 채점 기준을 최우선으로 따른다.
- 불확실하면 억지로 맞히지 않고 `needsReview: true`로 반환하게 한다.
- 학생에게 보여줄 피드백은 짧고 구체적인 한국어로 만든다.

## 구현 우선순위

1. 현재 OCR 결과를 `SubmissionPayload` JSON으로 변환
2. `MockAIGrader`로 발표용 AI 채점 결과 생성
3. Streamlit 화면에 AI 결과 JSON 표시 또는 다운로드 추가
4. 개인 AWS 계정에서 Bedrock model access 활성화
5. `BedrockConverseGrader` smoke test 실행
6. Bedrock에 문항 crop 이미지 전달
7. 서술형 샘플 1개를 Claude 검토 대상으로 시연
8. 재채점 프롬프트와 응답 포맷 추가
9. 오답 유형 집계용 간단한 Analytics JSON 설계

## 다음에 정해야 할 것

- 사용할 Bedrock 모델 ID
- 객관식/단답형 위주로 안정성을 잡을지, 서술형 1개를 무리해서라도 넣을지
- 백엔드 팀에 넘길 endpoint 이름과 JSON 필드명을 우리 안으로 확정할지
- 샘플 문제 이미지를 몇 장까지 확보할 수 있는지
