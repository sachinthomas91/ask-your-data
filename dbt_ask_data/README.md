# dbt_ask_data

A comprehensive dbt project for transforming and analyzing Olist e-commerce platform data. This project follows dbt best practices with a multi-layered architecture for data transformation.

## Project Overview

This dbt project contains transformation models for the Olist dataset, including customer behavior, seller performance, product analytics, and marketing insights. The project is designed to support data-driven decision making and natural language queries on business data.

## Project Structure

The project is organized into three main layers:

### Staging (`models/staging/olist/`)
Raw data cleaning and standardization layer. Creates views of source data with minimal transformations.
- Customer, order, and product staging
- Review and payment staging
- Geolocation and category data

### Intermediate (`models/intermediate/olist/`)
Business logic and aggregations layer. Creates tables with dimensional and factual data.
- **Dimensions**: `dim_customers`, `dim_products`, `dim_sellers`, `dim_date`, `dim_location`
- **Facts**: `fact_orders`, `fact_order_lines`, `fact_payments`, `fact_reviews`, `fact_closed_deals`
- **Aggregations**: Daily sales metrics, product-seller relationships

### Analytics/Marts (`models/marts/analytics/olist/`)
Analytics-ready views for reporting and insights.
- Customer segmentation and churn risk analysis
- Seller and customer performance metrics
- Product quality and repurchase metrics
- Marketing channel performance analysis

## Getting Started

### Prerequisites
- dbt installed (version compatible with your dbt Cloud/CLI setup)
- Database connection configured in `profiles.yml`

### Quick Start

1. Install dependencies:
   ```
   dbt deps
   ```

2. Run all models:
   ```
   dbt run
   ```

3. Run tests:
   ```
   dbt test
   ```

4. Generate documentation:
   ```
   dbt docs generate
   dbt docs serve
   ```

## Key Features

- **Modular architecture** with clear separation of concerns
- **Data quality tests** on critical models
- **Comprehensive documentation** with column-level descriptions
- **Schema organization** with staging, intermediate, and analytics schemas
- **Seed data** for reference tables and static data

## Configuration

Models are configured in `dbt_project.yml` with:
- Staging models materialized as views
- Intermediate models materialized as tables
- Analytics models materialized as views for optimal query performance

## Resources

- [dbt Documentation](https://docs.getdbt.com/docs/introduction)
- [dbt Community](https://community.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
- [Olist Dataset Documentation](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
