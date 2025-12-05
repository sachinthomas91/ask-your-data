# Ask Your Data - Olist E-commerce & Marketing Analytics

This dbt project transforms the [Olist Brazilian E-commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and the [Marketing Funnel Dataset](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) into analytical data marts. It is designed to power the "Ask Your Data" natural language query application.

## 📊 Project Overview

The goal of this project is to provide clean, aggregated, and business-ready data for analyzing:
- **Seller Performance**: Sales volume, revenue, and delivery performance.
- **Customer Behavior**: Purchase frequency, monetary value, and churn risk.
- **Sales Trends**: Daily and monthly revenue tracking.
- **Marketing ROI**: Lead conversion rates, channel performance, and deal values.

## 🏗️ Key Data Models

### Analytics Marts (`models/marts/analytics/olist`)

*   **`agg_seller_performance`**: The core seller dimension. Contains aggregated metrics for every seller, including:
    *   `total_orders`, `total_revenue`
    *   `avg_delivery_delay_days`
    *   `seller_status` (Active, Churned, etc.)
    *   `dominant_category`

*   **`agg_customer_performance`**: Customer-level analytics focusing on RFM (Recency, Frequency, Monetary) metrics:
    *   `total_spent`, `order_count`
    *   `days_since_last_purchase`
    *   `customer_segment` (VIP, Regular, New, etc.)

*   **`agg_sales_daily`**: A daily time-series of sales performance, useful for trend analysis and forecasting.

*   **`agg_marketing_channel_performance`**: Marketing funnel analysis linking MQLs (Marketing Qualified Leads) to closed deals.
    *   `mql_count`, `deals_closed`
    *   `mql_to_deal_conversion_rate`
    *   `avg_days_to_close`, `avg_deal_value`
    *   `channel_roi`, `quality_score`

*   **`agg_seller_health_scorecard`**: A composite scoring model that rates sellers based on sales, delivery speed, and review scores.

## ⚙️ Technical Architecture

### Handling Stale Data (Static Dataset)
The Olist dataset covers the period from **2016 to 2018**. To make this static historical data useful for "current" analysis (e.g., calculating "days since last sale"), we implemented a **Stale Date Handling** strategy.

Instead of using `CURRENT_DATE` (which would make all customers look churned in 2025), we use a dynamic reference date based on the maximum order date in the dataset.

**Logic:**
1.  A CTE `global_metrics` calculates `max_date` (the latest `order_purchase_timestamp` in the entire dataset).
2.  All "recency" calculations (e.g., `days_since_last_purchase`) are calculated relative to this `max_date`.

```sql
-- Example Pattern
WITH global_metrics AS (
    SELECT MAX(order_purchase_timestamp)::DATE AS max_date 
    FROM {{ ref('stg_olist__orders') }}
)
SELECT
    seller_id,
    (gm.max_date - last_sale_date) AS days_since_last_sale
FROM ...
CROSS JOIN global_metrics gm
```

This ensures that the analytics reflect the state of the business *as of the data's end date*, providing meaningful insights regardless of when the query is run.

## 🚀 Getting Started

### Prerequisites
- dbt Core installed
- Postgres database with Olist raw data loaded

### Running the Project

1.  **Install Dependencies**:
    ```bash
    dbt deps
    ```

2.  **Build Models**:
    ```bash
    dbt run
    ```

3.  **Test Data Quality**:
    ```bash
    dbt test
    ```

### Resources
- Learn more about dbt [in the docs](https://docs.getdbt.com/docs/introduction)
- Check out [Discourse](https://discourse.getdbt.com/) for commonly asked questions and answers
