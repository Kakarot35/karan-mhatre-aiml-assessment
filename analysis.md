# Lead Conversion Prediction — Exploratory Data Analysis

## Overview

The objective of this analysis is to understand the behavioral and firmographic characteristics of leads and identify factors that influence conversion probability.

The analysis was performed using two datasets:

* `leads.csv` — lead-level attributes
* `interactions.csv` — behavioral interaction logs

EDA was conducted to:

* Understand data quality
* Analyze conversion behavior
* Identify key engagement signals
* Discover patterns across lead sources, segments, industries, and funnel stages
* Support feature engineering for model training

---

# Data Overview

**Conversion Rate:** ~18% (derived from behavioral interaction signals)

### Data Types

The datasets contained a mix of:

- Numerical variables (`employee_count`, `click_count`, `session_duration_seconds`)
- Categorical variables (`source`, `company_size`, `lead_segment`, `industry`)
- Datetime variables (`created_at`, `timestamp`)


## Leads Dataset

**Shape:** `(2045, 21)`

The leads dataset contains firmographic and demographic information for each lead.

### Important Columns

| Column           | Description                   |
| ---------------- | ----------------------------- |
| `lead_id`        | Unique identifier             |
| `source`         | Acquisition channel           |
| `industry`       | Industry category             |
| `company_size`   | Company size                  |
| `lead_segment`   | Business segment              |
| `job_role`       | Lead designation              |
| `employee_count` | Number of employees           |
| `funding_stage`  | Startup/company funding stage |
| `created_at`     | Lead creation timestamp       |

---

## Interactions Dataset

**Shape:** `(40000, 36)`

The interactions dataset contains behavioral engagement data.

### Important Columns

| Column                     | Description            |
| -------------------------- | ---------------------- |
| `interaction_id`           | Unique interaction ID  |
| `lead_id`                  | Foreign key to lead    |
| `session_id`               | User session           |
| `timestamp`                | Event timestamp        |
| `event_name`               | Interaction event      |
| `page_name`                | Page visited           |
| `session_duration_seconds` | Time spent in session  |
| `scroll_depth_percent`     | Scroll engagement      |
| `click_count`              | User interactions      |
| `funnel_stage`             | Customer journey stage |

---

## Missing Value Analysis

Missing value analysis was conducted to identify incomplete records.

### Findings

The following columns contained missing values:

| Column                | Observation            |
| --------------------- | ---------------------- |
| `browser`             | Missing values present |
| `company_size`        | Missing values present |
| `annual_revenue_band` | Missing values present |
| `city`                | Missing values present |

### Interpretation

Missing values were relatively limited and did not significantly impact overall dataset usability.

For model training:

* Missing categorical values were handled appropriately.
* No severe missing-data issue was observed.

---

# Univariate Analysis

Univariate analysis was performed to understand the distribution of individual variables.

### Employee Count

The distribution of `employee_count` was heavily right-skewed, with a small number of very large organizations.

### Session Duration

Most sessions were short in duration, while a small subset of users demonstrated significantly longer engagement times.

### Funnel Stage

The majority of interactions occurred in earlier funnel stages, while only a smaller portion progressed to later decision-making stages.

### Lead Source

Google, LinkedIn, and Email Campaigns contributed the highest number of leads.

---

# Bivariate Analysis

## Conversion Distribution

The dataset shows class imbalance between converted and non-converted leads.

### Findings

* Non-converted leads represent the majority.
* Converted leads represent a smaller percentage of the dataset.

### Interpretation

This imbalance indicates a realistic business setting where only a fraction of leads become customers.

This justified the use of:

* F1-score
* Precision
* Recall
* AUC-ROC

instead of relying only on accuracy.

---

## Conversion Rate by Lead Source

Lead acquisition source significantly influenced conversion probability.

### Key Findings

| Source         | Observation                   |
| -------------- | ----------------------------- |
| LinkedIn       | Highest conversion rate       |
| Google         | Strong conversion performance |
| Email Campaign | Moderate performance          |
| Instagram      | Lowest conversion rate        |

### Business Insight

High-performing channels such as **LinkedIn** and **Google** should receive increased marketing focus.

Lower-performing channels like **Instagram** may require strategy changes or reduced spending.

---

## Conversion Rate by Lead Segment

Conversion rates varied across lead segments.

### Findings

| Segment    | Observation         |
| ---------- | ------------------- |
| Enterprise | Highest conversion  |
| Mid-Market | Strong performance  |
| Startup    | Moderate conversion |
| SMB        | Lowest conversion   |

### Business Insight

Enterprise leads appear more likely to convert and may deserve higher prioritization by sales teams.

---

## Conversion Rate by Industry

Different industries showed different conversion behavior.

### Findings

Higher conversion industries included:

* BFSI
* Technology
* Healthcare

Lower conversion industries included:

* Education
* Consulting

### Business Insight

Industry-specific sales strategies may improve conversion efficiency.

---

## Conversion Rate by Job Role

Lead designation influenced conversion behavior.

### Findings

Higher conversion roles:

* Analyst
* Manager
* Senior Manager

Lower conversion roles:

* Associate
* Intern

### Business Insight

Decision-making roles showed stronger buying intent compared to junior positions.

---

## Conversion Rate by Funding Stage

Funding maturity influenced conversion likelihood.

### Findings

Companies in:

* Public
* Series C+
* Series A

showed stronger conversion performance compared to:

* Seed-stage companies

### Business Insight

More mature companies may have greater purchasing power and readiness.

---

## Funnel Stage Analysis

Funnel progression strongly impacted conversion probability.

### Findings

Leads deeper in the funnel showed significantly higher conversion rates.

* Stage 1 → Very low conversion
* Stage 2 → Moderate conversion
* Stage 3 → High conversion
* Stage 4 → Highest conversion

### Business Insight

Funnel stage is one of the strongest signals of buying intent.

Leads reaching later funnel stages should receive immediate sales attention.

---

## Session Count vs Conversion

Session behavior showed strong separation between converters and non-converters.

### Findings

Converted leads had:

* Higher session counts
* Greater engagement
* More repeated visits

Non-converted users typically had fewer sessions.

### Business Insight

Repeat engagement is a strong conversion signal.

Users with multiple sessions should be prioritized.

---

## Correlation Analysis

A correlation heatmap was generated for numeric variables.

### Findings

Strong relationships observed:

* `session_count` → positively correlated with conversion
* `company_age_years` → mild positive relationship
* `employee_count` → weak relationship

### Interpretation

Behavioral engagement variables appear more predictive than static company information.

---

# Behavioral Segments Discovered

Based on engagement behavior, four informal lead segments were identified:

| Segment         | Characteristics                                            | Conversion Trend  |
| --------------- | ---------------------------------------------------------- | ----------------- |
| High Intent     | Multiple sessions, pricing visits, deep funnel progression | High conversion   |
| Researchers     | High page exploration, moderate engagement                 | Medium conversion |
| Window Shoppers | Low engagement, few sessions                               | Low conversion    |
| Fast Converters | Quick funnel progression                                   | High conversion   |

### Key Insight

Behavioral patterns strongly separate converters from non-converters.

Leads demonstrating repeat engagement and deeper funnel progression consistently showed stronger purchase intent.

---

# Anomalies & Data Quality Issues

Several data quality checks were performed.

### Observations

* Missing values were present in selected categorical columns.
* Outliers existed in `employee_count`.
* Certain acquisition channels underperformed significantly.
* Some companies had unusually large employee counts.

### Resolution

* Missing values were handled during preprocessing.
* Outliers were analyzed but retained since they represent legitimate business entities.
* No severe data corruption issues were identified.

---

## Outlier Analysis

Outlier detection was performed on `employee_count`.

### Findings

Extreme outliers were identified:

* Very large organizations with unusually high employee counts.

A log transformation visualization was also used to better understand the distribution.

### Interpretation

The presence of large enterprise organizations introduced skewness into the dataset.

Outliers were analyzed but retained since they represent real businesses.

---

# Temporal Patterns

Time-based analysis was performed to understand behavioral trends.

### Findings

* Certain months showed increased interaction activity.
* Demo requests fluctuated across time periods.
* Lead engagement changed depending on seasonality.
* No strong seasonal anomalies were observed, although interaction volume fluctuated across periods.

### Business Insight

Marketing and sales campaigns may benefit from aligning with high-engagement periods.

Understanding temporal behavior can improve campaign effectiveness.

---

# Key Insights

The EDA revealed several important business patterns:

1. **Lead source strongly affects conversion**
   LinkedIn and Google performed better than Instagram.

2. **Enterprise leads convert more frequently**
   Large organizations showed higher purchase intent.

3. **Funnel stage is highly predictive**
   Later funnel stages strongly correlate with conversion.

4. **Session count matters**
   Returning users are more likely to convert.

5. **Behavioral signals outperform static information**
   Engagement metrics provide stronger predictive power than demographic features.

---

# Business Recommendations

Based on the analysis, the following recommendations are suggested:

### 1. Prioritize Enterprise Leads

Enterprise leads consistently demonstrated stronger conversion behavior and should receive higher sales priority.

### 2. Focus on High-Performing Channels

Increase investment in:

* LinkedIn
* Google

Reduce inefficient spending on low-performing channels.

### 3. Prioritize High Engagement Users

Leads with:

* Multiple sessions
* Pricing page visits
* Deep funnel progression

should be routed quickly to sales teams.

### 4. Build Segment-Specific Strategies

Different lead segments behave differently and may require customized messaging.

### 5. Improve Lead Scoring

Behavioral engagement metrics should receive higher weight than demographic information in lead prioritization systems.

---

# Conclusion

The exploratory analysis identified meaningful behavioral and business patterns that informed feature engineering and model selection.

The strongest indicators of conversion were:

* Funnel stage progression
* Session count
* Time spent
* Lead source
* Company segment
* Engagement activity

These findings directly guided the development of the Lead Conversion Prediction model and the FastAPI inference system.
