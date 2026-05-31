# Mock Lambda deploy

This path deploys the CheckMate grader as a zip Lambda for end-to-end BE -> Lambda -> BE callback testing.

It is intentionally mock-only:

- Handler: `ai_grading.lambda_handler.handler`
- Runtime: `python3.12`
- Region: `us-east-1`
- Env: `AI_GRADING_MODE=mock`
- Env: `APP_INTERNAL_TOKEN=change-me-in-prod`
- No Bedrock, OCR, OpenCV, Torch, or Transformers dependencies are packaged.

## Prerequisites

Install and configure AWS CLI credentials outside this repository.

The first deploy needs an IAM role ARN for Lambda execution. The role only needs basic Lambda log permissions for mock mode, for example the AWS managed `AWSLambdaBasicExecutionRole` policy.

Current mock execution role:

```text
arn:aws:iam::259808926999:role/Checkmate-lambda-execution-role
```

## Build the zip

```powershell
.\scripts\package_lambda_mock.ps1
```

The zip is written to:

```text
build/checkmate-ai-grading-mock.zip
```

The package script asks pip for Linux-compatible wheels with `manylinux2014_x86_64`, so do not replace it with a normal Windows `pip install -t` when deploying to Lambda.

## First deploy

```powershell
.\scripts\deploy_lambda_mock.ps1 `
  -FunctionName checkmate-ai-grading-mock `
  -Region us-east-1 `
  -RoleArn arn:aws:iam::259808926999:role/Checkmate-lambda-execution-role `
  -AppInternalToken change-me-in-prod
```

## Update an existing function

```powershell
.\scripts\deploy_lambda_mock.ps1 `
  -FunctionName checkmate-ai-grading-mock `
  -Region us-east-1 `
  -AppInternalToken change-me-in-prod
```

## BE settings

Use the deployed BE URL for callbacks. Lambda cannot call a local `localhost:8080`.

```yaml
app:
  grading:
    callback-base-url: https://deployed-be.example.com

cloud:
  aws:
    lambda:
      grading-function-name: checkmate-ai-grading-mock
```

For local demos, expose the BE with ngrok or a similar tunnel and use that HTTPS URL as `callback-base-url`.

## Expected verification

After the BE invokes this Lambda on submission confirm, the callback should post a `DONE` result back to:

```text
/api/internal/submissions/{submissionId}/result
```

Every question receives full score in mock mode. Switch to a container-image Lambda before using `AI_GRADING_MODE=bedrock`.
