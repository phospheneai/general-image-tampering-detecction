#!/usr/bin/env bash
# One-time AWS setup for the training pipeline. Idempotent — safe to re-run.
# Run it once with admin credentials for the target account:
#   bash infra/setup-aws.sh
#
# Creates (in AWS_REGION, default us-east-1 — where the data bucket lives):
#   1. the GitHub OIDC provider        (so Actions needs no access keys)
#   2. role forgery-github-actions-role (Actions: push to ECR, deploy + start the pipeline)
#   3. role forgery-stepfunctions-role  (Step Functions: create training jobs)
#   4. role forgery-sagemaker-role      (the training job: pull the image, read data, write checkpoints)
#   5. the ECR repository forgery-train
# Prints the one value to paste into GitHub at the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "[setup] account $AWS_ACCOUNT_ID, region $AWS_REGION"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
render () { bash render.sh "$1" > "$TMP/$(basename "$1")"; echo "file://$TMP/$(basename "$1")"; }

# ── 1. GitHub OIDC provider ───────────────────────────────────────────────────
# GitHub signs a short-lived token for each workflow run; AWS trusts it for
# this repo only (see iam/github-trust.json). No stored access keys.
OIDC_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
  echo "[setup] OIDC provider already exists"
else
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com >/dev/null
  echo "[setup] OIDC provider created"
fi

# ── 2-4. roles ────────────────────────────────────────────────────────────────
make_role () {                       # name  trust-file  policy-file
  local trust policy
  trust=$(render "$2"); policy=$(render "$3")
  if aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$1" --policy-document "$trust"
    echo "[setup] role $1 exists — trust policy refreshed"
  else
    aws iam create-role --role-name "$1" --assume-role-policy-document "$trust" >/dev/null
    echo "[setup] role $1 created"
  fi
  aws iam put-role-policy --role-name "$1" --policy-name "$1-policy" --policy-document "$policy"
}
make_role forgery-github-actions-role iam/github-trust.json        iam/github-policy.json
make_role forgery-stepfunctions-role  iam/stepfunctions-trust.json iam/stepfunctions-policy.json
make_role forgery-sagemaker-role      iam/sagemaker-trust.json     iam/sagemaker-policy.json

# ── 5. ECR repository ─────────────────────────────────────────────────────────
aws ecr describe-repositories --repository-names forgery-train --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name forgery-train --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null
echo "[setup] ECR repository forgery-train ready"

cat <<MSG

──────────────────────────────────────────────────────────────────────────────
Done. Add ONE repository secret in GitHub — no access keys anywhere:

  Settings > Secrets and variables > Actions > New repository secret
    Name:  AWS_ROLE_ARN
    Value: arn:aws:iam::${AWS_ACCOUNT_ID}:role/forgery-github-actions-role

Before the first smoke test, upload (once):
  # the DINOv3 ViT-L/16 backbone (needs HF access to the gated model)
  bash scripts/download_artifacts.sh
  aws s3 cp --recursive artifacts/mirror/dinov3-vitl16/ \
      s3://authenta-data-rnd/image-tampering-detection/artifacts/dinov3-vitl16/ \
      --exclude '*' --include config.json --include model.safetensors
  # the smoke slice of processed-v1
  python scripts/make_smoke_mds.py --upload

Then run the workflows, in order:
  1. "Build training image"  -> builds and pushes to ECR
  2. "Deploy pipelines"      -> creates forgery-training AND forgery-smoke
  3. "Smoke test"            -> 1 epoch on the smoke slice, waits, red/green
  4. "Run training"          -> the real run
──────────────────────────────────────────────────────────────────────────────
MSG
