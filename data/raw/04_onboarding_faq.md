# Nimbus Cloud — Onboarding FAQ

**How long does initial setup take?**
Most customers on the Starter or Growth plan complete their first
working pipeline within 30 minutes using the guided setup wizard.
Enterprise customers with self-hosted or hybrid deployments typically
take 1-2 weeks, largely due to VPC networking configuration on the
customer's side.

**Do I need to write any code?**
No. The visual pipeline builder covers the vast majority of use cases
without code. SQL is only needed if you want custom transform logic
beyond the built-in transform blocks (deduplication, type casting,
field mapping, filtering).

**Can I test a pipeline before it goes live?**
Yes. Every pipeline has a "dry run" mode that processes a sample of up
to 1,000 rows without writing to the destination, so you can validate
the output schema and transform logic first.

**What happens if a sync fails partway through?**
Nimbus uses checkpointing, so a failed sync resumes from the last
successful checkpoint rather than restarting from scratch. Failed rows
are logged individually and can be reprocessed after the underlying
issue is fixed, without re-running the entire pipeline.

**Can I migrate from another ETL tool?**
Nimbus provides a migration assistant for customers coming from Fivetran
or Airbyte, which can import existing connector configurations
automatically. Custom or less common source tools may require manual
reconfiguration.

**Who do I contact for onboarding help?**
Growth plan customers get one free onboarding call with a Nimbus
solutions engineer. Enterprise customers are assigned a dedicated
solutions engineer for the full onboarding period. Starter plan
customers use the self-serve documentation and community forum.
