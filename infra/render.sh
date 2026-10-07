#!/usr/bin/env bash
# Print an infra template (IAM policy or ASL) with the account and region filled
# in. The templates stay account-free in git; everything that sends them to AWS
# (setup-aws.sh, the Deploy pipelines workflow) goes through this.
#
#   AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1 bash infra/render.sh infra/train-pipeline.asl.json
set -euo pipefail
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${AWS_REGION:?set AWS_REGION}"
[[ "$AWS_ACCOUNT_ID" =~ ^[0-9]{12}$ ]] || { echo "bad AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID" >&2; exit 1; }
[[ "$AWS_REGION" =~ ^[a-z]{2}(-[a-z]+)+-[0-9]$ ]] || { echo "bad AWS_REGION: $AWS_REGION" >&2; exit 1; }

out=$(sed -e "s/\${AWS_ACCOUNT_ID}/${AWS_ACCOUNT_ID}/g" -e "s/\${AWS_REGION}/${AWS_REGION}/g" "$1")
# a placeholder nobody defined would otherwise reach AWS as a literal string
if grep -q '\${' <<<"$out"; then
  echo "unrendered placeholder left in $1:" >&2
  grep -n '\${' <<<"$out" >&2
  exit 1
fi
printf '%s\n' "$out"
