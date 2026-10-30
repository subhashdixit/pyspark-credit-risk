# Data provenance

Serving/training source: Yeh, I. (2009). **Default of Credit Card Clients**. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C55S3H . Official source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients . Licensed **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ .

Adaptations: select 19 financial-history features, exclude ID/sex/education/marital-status/age, group identical selected profiles, fit training-only median imputation and scaling, train Spark logistic regression, export numeric parameters. The source spreadsheet is downloaded unchanged and ignored by Git; its SHA-256 is recorded in models/metrics.json. This historical Taiwan credit-card dataset is different from the original notebook's Home Credit competition dataset. Original notebook/PDF remain under Codes/.

Omitting demographic fields does not establish fairness: financial variables can still be proxies. Currency amounts are source NTD, not INR or USD. Source months run September to April 2005; the target is next-month default. Repayment code -1 is documented as paid duly; source -2 and 0 codes are retained without inventing a new semantic meaning.
