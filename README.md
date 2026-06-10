# AI/ML Engineer Assessment - Lead Conversion Prediction

##  Project Overview

This project focuses on building an end-to-end machine learning pipeline to predict the probability of a lead converting into a paying customer. By analyzing behavioral interactions (such as pages visited, time spent, and feature usage) alongside static lead profile data (such as company size and acquisition channel), the system generates highly accurate probability scores and business-readable explanations.

The system includes:
* **Exploratory Data Analysis (EDA):** Deep dives into behavioral metrics to uncover high-intent signals.
* **Feature Engineering:** Transforming raw event logs into actionable lead-level aggregations.
* **Machine Learning Model Training:** Evaluating multiple algorithms to find the most robust predictive model.
* **FastAPI Inference API:** Exposing the model via a high-performance RESTful service.
* **Rule-Based Prediction Explanation System:** Generating transparent, human-readable summaries that explain *why* a lead received a certain score, empowering sales representatives with actionable context.

The ultimate objective is to help sales and marketing teams identify high-intent leads, prioritize their outreach efforts, optimize resource allocation, and ultimately accelerate the sales cycle.

---

## Project Structure

The repository is modularized for clarity and maintainability:

```txt
karan-mhatre-aiml-assessment/
├── README.md                 # Project documentation and setup guide
├── requirements.txt          # Python dependencies required to run the project
├── train.py                  # End-to-end pipeline script for training the ML model
├── app.py                    # FastAPI application serving the inference and explanation endpoints
├── config.py                 # Centralized configuration for hyperparameters, file paths, and API settings
├── utils.py                  # Helper functions for data loading, logging, and missing value imputation
├── notebooks/                # Directory for exploratory Jupyter notebooks
│   └── eda.ipynb             # Comprehensive EDA, visualization, and statistical analysis
├── outputs/                  # Auto-generated directory for pipeline artifacts
│   ├── model.pkl             # Serialized best-performing ML pipeline
│   ├── model_metrics.json    # JSON report of classification metrics from the latest run
│   ├── feature_importance.png# Visual plot of the most influential features
│   └── eda/                  # Directory for saved EDA visualizations
├── .gitignore                # Specifies intentionally untracked files to ignore
```

---

##  Setup Instructions

### 1. Clone the Repository

Begin by cloning the project to your local machine:
```bash
git clone <your-github-repo-url>
cd karan-mhatre-aiml-assessment
```

### 2. Create a Virtual Environment

It is highly recommended to run this project in an isolated Python environment (Python 3.9+ is recommended).

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using `pip`. This will install essential libraries like `pandas`, `scikit-learn`, `fastapi`, and `uvicorn`.
```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Because the dataset contains sensitive assessment data, it is excluded from version control via `.gitignore`. You must manually place the provided CSV files inside the `data/` directory before running the training script.

```txt
data/
├── leads.csv          # Contains static lead metadata (e.g., source, company size)
├── interactions.csv   # Contains granular event logs (e.g., page views, sessions, durations)
```

Ensure the column names in your CSVs match the expectations set in `config.py` and the feature engineering steps.

---

## Exploratory Data Analysis (EDA)

The EDA phase was critical for understanding the underlying distribution of the data and identifying the signals that correlate most strongly with a successful conversion. The analysis covers:

* **Data quality and missing values:** Identifying nulls and deciding on imputation strategies (median for numeric, mode for categorical).
* **Conversion patterns:** Visualizing the overall conversion rate to understand class imbalance.
* **Lead source effectiveness:** Grouping conversions by channel to see which marketing campaigns yield the highest ROI.
* **Funnel stage impact:** Analyzing how leads progress from awareness to decision.
* **Industry and Segment performance:** Evaluating which company sizes and business segments are our strongest product-market fit.
* **Outliers and anomalies:** Detecting bots or idle sessions with abnormally high interaction durations.

You can view the full analysis in the interactive notebook:
```txt
notebooks/eda.ipynb
```

### Key Findings & Insights
* **LinkedIn and Google** generated the strongest conversion rates, indicating that search intent and professional networking are high-value acquisition channels.
* **Enterprise leads** showed better conversion performance than SMB leads, suggesting a strong product-market fit for larger organizations.
* Leads in **deeper funnel stages** converted significantly more often, proving that funnel progression is a reliable indicator of intent.
* **Session count and engagement behavior** (like pages visited and total time spent) strongly influenced conversion probability. High interaction volume is the single strongest predictor.
* **Pricing page interactions** were strongly associated with higher buying intent, serving as a clear bottom-of-the-funnel signal.

---

## Model Training

To ensure we use the most accurate predictive model, multiple algorithms were evaluated during the training phase:

* **Logistic Regression:** Used as a baseline for its interpretability.
* **Random Forest:** An ensemble method that handles non-linear relationships and feature interactions well.
* **Gradient Boosting & XGBoost:** Advanced boosting techniques that sequentially correct errors.

### Best Performing Model: **Random Forest Classifier**
The Random Forest model was selected because it offered the best balance of high recall (catching potential converters) and high precision (not overwhelming sales with false positives), while remaining robust against overfitting.

### Performance Metrics
| Metric    | Score  | Description |
| --------- | ------ | ----------- |
| **Accuracy**  | 0.8435 | Overall correctness of the model predictions. |
| **Precision** | 0.7343 | When the model predicts a conversion, it is correct 73.4% of the time. |
| **Recall**    | 0.8015 | The model successfully identifies 80.1% of all actual conversions. |
| **F1 Score**  | 0.7664 | The harmonic mean of precision and recall. |
| **AUC-ROC**   | 0.9172 | Excellent ability to distinguish between converters and non-converters. |

### How to Train the Model
To execute the data preprocessing, feature engineering, and model training pipeline, simply run:
```bash
python train.py
```

This will process the data and generate the following artifacts:
```txt
outputs/
├── model.pkl                  # The saved pipeline used by the FastAPI service
├── model_metrics.json         # The metrics table shown above
├── feature_importance.png     # A bar chart showing the top 15 most predictive features
```

---

##  Running the API

The project uses FastAPI to serve the model as a highly concurrent REST API.

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

FastAPI automatically generates interactive API documentation. Once the server is running, you can explore and test the endpoints directly from your browser:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **ReDoc UI:** `http://127.0.0.1:8000/redoc`

---

## 🔌 API Endpoint Documentation

### 1. Health Check
Used to verify that the API is running and that the serialized model has been successfully loaded into memory.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": "true"
}
```

---

### 2. Predict Lead Conversion
Accepts lead profile and behavioral data, runs it through the loaded Random Forest pipeline, and returns a conversion probability along with business-friendly categorization (confidence and risk levels).

**Endpoint:** `POST /predict`

**Example Request:**
```json
{
  "pages_visited": 14,
  "time_spent_minutes": 45.0,
  "pricing_views": 3,
  "session_count": 4,
  "days_since_first_visit": 12,
  "first_touch_channel": "Google",
  "company_size": "Medium"
}
```

**Example Response:**
```json
{
  "lead_id": "ad-hoc-lead",
  "conversion_probability": 0.163,
  "confidence": "low",
  "risk_level": "high"
}
```
*Note: A `high` risk level means there is a high risk that the lead will **not** convert, whereas a `low` confidence means the model is fairly confident this is not a strong prospect.*

---

### 3. Explain Prediction
Translates the numerical prediction and raw features into a concise, human-readable summary. This is designed to be displayed directly in a CRM so a sales rep knows exactly *why* a lead is highly rated.

**Endpoint:** `POST /explain`

**Example Request:**
```json
{
  "conversion_probability": 0.84,
  "pages_visited": 14,
  "demo_requests": 1,
  "pricing_views": 3
}
```

**Example Response:**
```json
{
  "summary": "This lead shows high conversion intent because the behavior includes a demo request, multiple pricing page visits, and broad product exploration across many pages. Sales should prioritize timely follow-up with messaging focused on product fit and next-step clarity."
}
```

---

## Feature Engineering

A significant portion of this project involved aggregating raw time-series event logs into structured, lead-level behavioral summaries.

**Behavioral Features (Engineered from interactions):**
* **Pages Visited:** Total volume of page views per lead.
* **Time Spent:** Cumulative duration of all sessions in minutes.
* **Pricing Page Views:** A targeted count of high-intent URL visits.
* **Session Count:** Total number of distinct active sessions.
* **Days Since First Visit:** Time elapsed between the first interaction and the lead creation timestamp (measures sales cycle length).
* **Scroll Depth:** Average percentage of the page scrolled across all sessions.
* **Unique Pages:** Breadth of product exploration.
* **Active Days:** Total number of distinct days the lead interacted with the site.

**Categorical Features (From lead profile):**
* **First Touch Channel:** The marketing source (e.g., Google, LinkedIn, Organic).
* **Company Size:** e.g., SMB, Mid-Market, Enterprise.
* **Lead Segment:** The assigned industry or business category.
* **Device Type:** Mobile, Desktop, Tablet.

---

## Limitations

* **Dataset Size:** The model was trained on a localized, static assessment dataset. A larger, more dynamic dataset would improve generalization.
* **Rule-Based Explanations:** The `/explain` endpoint currently uses static, deterministic rules. While fast and reliable, it lacks the nuanced conversational capability of a true generative AI.
* **Data Drift:** Over time, user behavior and marketing channels change. This model's performance may degrade if it is not periodically retrained on fresh data.

---

##  Future Improvements

If this project were to be extended for a true production environment, the following improvements would be prioritized:

* **Dockerization:** Containerizing the FastAPI app and model using Docker to ensure environment consistency across deployments.
* **Cloud Deployment:** Hosting the application via AWS ECS, Google Cloud Run, or Azure App Service for infinite scalability.
* **LLM-Powered Explanations:** Replacing the static `/explain` endpoint with an integration to Gemini or OpenAI to dynamically generate highly contextual, personalized sales summaries.
* **Real-Time Streaming Pipeline:** Moving from batch CSV processing to a real-time event streaming architecture (e.g., Kafka) to instantly update lead scores as users navigate the website.
* **Model Monitoring:** Implementing tools like EvidentlyAI to track feature drift and automatically trigger retraining pipelines.
* **Hyperparameter Tuning:** Utilizing GridSearch or Optuna to squeeze the maximum possible performance out of the Random Forest estimator.

---

## Author

Developed by **Karan Mhatre** as part of the **Vynqe AI/ML Engineer Assessment**.
