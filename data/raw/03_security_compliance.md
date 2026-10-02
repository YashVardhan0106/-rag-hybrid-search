# Nimbus Cloud — Security & Compliance

## Certifications

Nimbus Cloud is SOC 2 Type II certified, with the most recent audit
completed in March 2025. The company is also GDPR compliant for
customers processing data from the EU, and offers a signed Data
Processing Addendum (DPA) on request.

Nimbus is not currently HIPAA certified. Customers in healthcare who
require HIPAA compliance should contact the sales team about the
Enterprise plan's dedicated compliance track, which is in beta.

## Data Encryption

All data is encrypted in transit using TLS 1.3. Data at rest is
encrypted using AES-256. Encryption keys are managed via AWS KMS, and
Enterprise customers may optionally bring their own encryption keys
(BYOK).

## Access Control

Nimbus supports role-based access control (RBAC) with four built-in
roles: Admin, Editor, Viewer, and Billing. Enterprise plans can define
custom roles with granular, per-connector permissions. Single sign-on
(SSO) via SAML 2.0 is available on Growth and Enterprise plans; Starter
plan users must use email/password or Google OAuth.

## Data Residency

By default, customer data is processed in US-East (Virginia) data
centers. EU customers on the Growth plan or higher can request that
their data be processed exclusively in EU-West (Frankfurt) to satisfy
data residency requirements.

## Incident Response

Nimbus maintains a documented incident response plan and commits to
notifying affected customers within 72 hours of confirming a security
incident that impacts their data, in line with GDPR breach notification
requirements. A public status page tracks uptime and ongoing incidents
in real time.
