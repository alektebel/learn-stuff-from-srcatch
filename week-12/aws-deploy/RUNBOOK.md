# RUNBOOK — a real account, a real deploy

**This is not graded and cannot be.** `check.py` lints artifacts offline;
everything here happens on your own AWS account with your own card attached.

Every command below should be verified against current AWS docs before you run
it — CLI syntax, service names and console flows change, and this was written
against a knowledge cutoff. Where something costs money it says so.

**The finish line, stated up front.** You control this when you can *rebuild
the entire stack from an empty account, from code, in under an hour, and tear
it back down to a zero bill.* Do that twice. Everything below is in service of
that one test.

---

## Phase 0 — before anything else · ~1 h · DO NOT SKIP

The two ways this goes wrong are a leaked credential and a forgotten resource.
Both are cheap to prevent now and expensive later.

- [ ] **Root account: enable MFA, then never use it again.** Root is for
      billing settings and closing the account. Nothing else.
- [ ] **Billing alarm before the first resource.** Budgets → a monthly budget at
      an amount you would notice (start low — $5), with an email alert at 50%,
      80% and 100% of *forecast*, not just actual. Forecast is the one that
      warns you in time.
- [ ] **Turn on Cost Explorer** and check it daily for the first fortnight. It
      lags ~24 h; that lag is why the alarm matters.
- [ ] **IAM Identity Center (SSO), not IAM users.** Create yourself an admin
      permission set and sign in through it. This is the single most important
      step: it gives you **short-lived** credentials by default.
- [ ] **No long-lived access keys. Ever.** If you find yourself creating one,
      stop and work out what you actually needed. `credentials.py`'s check
      exists because this is the mistake that is both expensive and public.
- [ ] **`~/.aws/` is not a repo.** Confirm your global gitignore covers it, and
      install a pre-commit secret scanner today rather than after.
- [ ] **Set a region and stick to it.** Resources in a region you forgot about
      are the classic forgotten-bill story.

**Done means:** a budget alarm you have *tested* by setting it to $0.01 and
receiving the mail, then set back.

---

## Phase 1 — the CLI and the SDK · ~3 h

- [ ] Install the AWS CLI v2. `aws configure sso`, then `aws sts get-caller-identity`
      — that one command tells you *who the CLI thinks you are*, and it is the
      first thing to run whenever something is mysteriously denied.
- [ ] `aws configure list` and understand the **credential resolution order**:
      CLI flags, env vars, profile, container role, instance role. Half of all
      "it works on my machine" AWS problems are this order.
- [ ] boto3: create a `Session`, get a client and a resource, and see the
      difference. Handle `ClientError` and read `.response["Error"]["Code"]` —
      the code is stable, the message is not.
- [ ] **Read-only tour.** `describe-*` and `list-*` across the eight services
      you built in week 1. Compare each real response to your toy version's
      shape. This is the hour where week 1 pays off.
- [ ] `assume-role` into a second permission set and watch the session expire.

**Done means:** you can answer "who am I, in which account, with what
permissions" in one command, and explain why a call failed from the error code
alone.

---

## Phase 2 — first deploy, by hand, then by code · ~4 h · costs pennies

- [ ] Static site: S3 bucket, upload with `aws s3 sync`, CloudFront in front,
      OAC so the bucket stays private. Do it **by CLI first** — you need to feel
      how many steps it is.
- [ ] Break it on purpose: wrong content type, missing index document, a
      CloudFront cache serving the old file. Fix each. The invalidation lesson
      is worth the whole phase.
- [ ] Now **delete all of it** and rebuild the identical thing from a
      CloudFormation or CDK template. Time both.
- [ ] `template.py`'s checks apply here: run your dependency ordering over your
      own template before you deploy it.
- [ ] Tear down. Confirm the bucket, the distribution and the OAC are all gone.

**Done means:** the second build is one command, and teardown leaves nothing in
the console.

---

## Phase 3 — Amplify, and what it is actually doing · ~4 h

Amplify is hosting + CI/CD + auth + API scaffolding, packaged. Its value *is*
the packaging, which is why building a toy version teaches nothing — but using
it while knowing what it wires up teaches a lot.

- [ ] Connect a git repo, add an `amplify.yml` build spec, and get a branch
      deploying on push.
- [ ] **Break the build on purpose** and read the log. Then fix it. The build
      log is the thing you will actually live in.
- [ ] Custom domain: point Route 53 (or your registrar) at it, wait out DNS,
      get the certificate issued. Budget an hour for propagation and do not
      spend it debugging.
- [ ] Branch previews and environment variables per branch.
- [ ] **Then go look at what it created for you** — the CloudFront
      distribution, the S3 bucket, the IAM service role. Compare that role's
      policy against `policy.py`'s least-privilege check. This is the phase's
      real lesson: Amplify made choices on your behalf and you should be able to
      name every one.

**Done means:** a push deploys, a broken push does not, and you can list what
Amplify provisioned and why.

---

## Phase 4 — a backend you own · ~6 h · watch the bill

- [ ] API Gateway (HTTP API — cheaper than REST for most cases) → Lambda →
      DynamoDB. All of it from IaC, none of it from the console.
- [ ] Cognito user pool, a JWT authorizer on the API, and a route that is
      genuinely rejected without a token. Confirm the rejection yourself.
- [ ] Lambda: set memory deliberately and measure — week 1's `optimize.py`
      showed a CPU-bound function costs the same at 128 MB and 10 GB. Verify it
      on the real thing.
- [ ] **Do not add a NAT gateway** unless you have proved you need one. It bills
      per hour whether or not anything uses it, and it is the classic surprise
      line. Use a VPC endpoint if you need private S3 or DynamoDB access —
      that's the crossover you already derived.
- [ ] Structured logs to CloudWatch, a metric filter, and one alarm that
      actually pages you.

**Done means:** an authenticated request round-trips, an unauthenticated one is
refused, and you can point at the log line for both.

---

## Phase 5 — operate it · ~3 h

- [ ] Deploy a bad version on purpose and roll back. Time the rollback.
- [ ] Add the health gate from `deploy.py`: promotion blocked until the new
      version serves the *right content*, not merely a 200.
- [ ] CloudWatch dashboard: p50/p99 latency, error rate, invocation count,
      throttles. Four numbers, one screen.
- [ ] Set an alarm on DynamoDB throttles and then **cause one** with a hot
      partition key — the failure you modelled in week 1.
- [ ] Read the bill line by line. Every line should be a resource you can name.

**Done means:** you have rolled back once for real, and no line on the bill
surprises you.

---

## Phase 6 — the test · ~2 h

- [ ] `terraform destroy` / `cdk destroy` / delete the stacks. Everything.
- [ ] Wait 24 h and check Cost Explorer reads zero for new spend.
- [ ] **Rebuild the whole thing from code, timed.** Target: under an hour, no
      console clicks, no manual steps.
- [ ] Tear down again.
- [ ] Do it once more, from a genuinely empty account if you can.

**Done means:** two clean rebuild-and-destroy cycles, and a repo that is the
only source of truth for what exists.

---

## Standing rules, after all of this

- **Nothing exists that is not in code.** A console click is a resource nobody
  will remember to delete.
- **Check the bill weekly.** Not monthly.
- **A credential on disk is an incident**, even on your own laptop, even
  briefly.
- **Tear down anything you are not actively using.** The free tier expires
  twelve months after you open the account, and things that were free become
  billable on a date you will not be watching for.

---

[Offline checks](check.py) · [TODO](../../TODO.md) · [Roadmap](../../ROADMAP.md)
