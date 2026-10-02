# Nimbus Cloud — Product Overview

Nimbus Cloud is a managed data-pipeline platform that lets teams move data
between databases, APIs, and warehouses without writing custom ETL code.
It was founded in 2021 and has processed over 4 billion records per month
as of its most recent usage report.

## Core Components

**Connectors** — Nimbus ships with 120+ pre-built connectors covering
databases (Postgres, MySQL, MongoDB), SaaS tools (Salesforce, HubSpot,
Stripe), and cloud warehouses (Snowflake, BigQuery, Redshift). Each
connector handles schema detection, incremental syncing, and automatic
retries on failure.

**Transform Layer** — Data can be transformed in-flight using either a
visual, no-code editor or raw SQL. Transforms run inside isolated
compute pods and scale horizontally based on data volume.

**Orchestration** — Pipelines can be scheduled (cron-based), triggered
by upstream events (webhooks, file-arrival), or run continuously in
streaming mode using a change-data-capture (CDC) engine.

**Observability** — Every pipeline run emits structured logs, row-level
error tracking, and latency metrics, all visible in the Nimbus dashboard.
Alerts can be routed to Slack, PagerDuty, or email.

## Deployment Models

Nimbus offers three deployment models:

1. **Cloud (SaaS)** — fully managed, hosted on Nimbus infrastructure.
   This is the default for new customers and requires no infrastructure
   management on the customer's part.
2. **Hybrid** — the control plane (scheduling, UI, monitoring) is hosted
   by Nimbus, but data processing runs inside the customer's own VPC.
   This model is popular with customers who cannot let raw data leave
   their network.
3. **Self-Hosted** — the entire stack, including the control plane, runs
   on customer infrastructure. This is only available on the Enterprise
   plan and requires a dedicated Nimbus solutions engineer for setup.

## Typical Use Cases

Nimbus is most commonly used for three workflows: syncing production
database tables into a warehouse for analytics, consolidating SaaS tool
data (marketing, sales, support) into a single reporting layer, and
replicating data across regions for disaster recovery.
