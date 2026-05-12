# Personal AWS Bedrock Setup

## Recommended Account Choice

Use a personal AWS account for the actual Bedrock demo. The AWS Academy account can remain useful for architecture screenshots, but the current Academy role is missing Bedrock permissions, so it is likely to block real model calls.

## Safety First

1. Open AWS Billing and create a budget.
2. Set alerts around USD 5 and USD 10 for this project.
3. Use one region for the demo. Start with `us-east-1` because Bedrock model availability is broad there.
4. Do not commit AWS credentials, `.env`, access keys, or CLI config files.

## Console Checklist

1. Sign in to the personal AWS account.
2. Switch region to `United States (N. Virginia) / us-east-1`.
3. Open Amazon Bedrock.
4. Go to `Model access` or `Model catalog`.
5. Choose one model for the first test.
   - Claude is preferred for math reasoning and Korean feedback.
   - Amazon Nova is a good fallback if third-party model access is slower.
6. If the console asks for use case details, fill it in for an education / grading assistant demo.
7. Wait a few minutes after enabling access.

## Local Environment

Install AWS dependencies:

```powershell
pip install -r requirements.txt
```

Configure credentials outside the repo:

```powershell
aws configure --profile checkmate
```

Set environment variables for the current shell:

```powershell
$env:AWS_PROFILE = "checkmate"
$env:AWS_REGION = "us-east-1"
$env:BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
```

For Claude Sonnet 4.6 in `us-east-1`, use the inference profile ID above. The base model ID `anthropic.claude-sonnet-4-6` can fail with an on-demand throughput error because this model is served through cross-region inference profiles in many regions.

## Minimal Python Smoke Test

After model access is enabled, run this from the project root:

```powershell
python -c "from ai_grader import BedrockConverseGrader, QuestionInput, dump_api_json, SubmissionPayload; q=QuestionInput(question_id='demo-1', problem_num=1, expected_answer='2', student_answer='2', ocr_confidence=0.95, ocr_source='manual', is_correct_by_rule=True); p=SubmissionPayload(submission_id='s-demo', assignment_id='a-demo', student_id='u-demo', created_at='local', questions=[q]); print(dump_api_json(BedrockConverseGrader().grade_submission(p)))"
```

Expected result: JSON with `source: "bedrock"` and a Korean grading reason.

Successful example:

```json
{
  "submissionId": "s-demo",
  "assignmentId": "a-demo",
  "studentId": "u-demo",
  "totalScore": 1.0,
  "maxScore": 1.0,
  "questions": [
    {
      "questionId": "demo-1",
      "problemNum": 1,
      "isCorrect": true,
      "score": 1.0,
      "maxScore": 1.0,
      "reason": "학생의 답안 '2'가 정답 '2'와 일치합니다.",
      "confidence": 0.99,
      "source": "bedrock",
      "needsReview": false,
      "deduction": ""
    }
  ]
}
```

## Streamlit Demo

The app keeps OCR/rule grading as the default to avoid accidental Bedrock charges. It also caches OCR results for the same uploaded page, so manual corrections and AI mode changes do not rerun the OCR pipeline every time.

1. Set `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `BEDROCK_MODEL_ID` in the PowerShell session that runs Streamlit.
2. Start or refresh the Streamlit app.
3. In the sidebar, choose an `AI 채점 방식`.
4. Upload a sample PDF/image and open `AI 채점 JSON`.

AI modes:

- `빠른 모드: OCR/rule`: no Bedrock call.
- `하이브리드: 불확실/서술형만 Claude`: calls Claude only for low-confidence, missing-answer, or descriptive questions.
- `서답형+서술형 Claude`: calls Claude for short-answer and descriptive questions.
- `전체 Claude`: calls Claude for every question and is the slowest/costliest mode.

The Streamlit path sends each question crop image to Claude along with OCR candidates, the expected answer, question type, and rubric. This enables:

- multiple choice: detect selected option from marks.
- short answer: find the final answer among mixed solution steps.
- descriptive: grade the solution process with a rubric.

## IAM Permissions For The Demo User

For a simple personal-account demo, attach the AWS managed `AmazonBedrockFullAccess` policy to the IAM user or role used for local testing. If you want a tighter custom policy later, the minimum runtime permissions are centered on:

- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:Converse`
- `bedrock:ConverseStream`

Some third-party models may also require one-time AWS Marketplace / model access permissions before invocation works.
