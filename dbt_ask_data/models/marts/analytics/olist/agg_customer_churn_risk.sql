-- Customer churn risk model
-- Grain: customer_id
-- Purpose: Identify customers at risk of churn and predict churn probability

with fact_orders as (
    select * from {{ ref('fact_orders') }}
)

, customer_purchase_recency as (
    select
        customer_id
        , max(order_purchase_datetime::date) as last_purchase_date
        , (current_date - max(order_purchase_datetime::date)) as days_since_last_purchase
        , min(order_purchase_datetime::date) as first_purchase_date
        , count(distinct order_id) as total_purchase_count
        , round(avg(order_total)::numeric, 2) as avg_order_value
        , round(sum(order_total)::numeric, 2) as total_lifetime_value
    from
        fact_orders
    where
        order_status not in ('Canceled', 'Unavailable')
    group by
        customer_id
)

, churn_risk_calculation as (
    select
        customer_id
        , first_purchase_date
        , last_purchase_date
        , days_since_last_purchase
        , total_purchase_count
        , avg_order_value
        , total_lifetime_value
        -- Calculate average days between purchases
        , case
            when total_purchase_count > 1
                then round(((last_purchase_date - first_purchase_date) / (total_purchase_count - 1))::numeric, 2)
            else null
          end as avg_days_between_purchases
        -- Churn score (0-100): Higher = Higher risk
        , round(
            (
              -- Recency factor (0-40 points): Penalize recent inactivity
              case
                when days_since_last_purchase <= 30 then 5
                when days_since_last_purchase <= 60 then 15
                when days_since_last_purchase <= 90 then 25
                when days_since_last_purchase <= 180 then 35
                else 40
              end
              +
              -- Frequency factor (0-30 points): Lower purchase frequency = Higher risk
              case
                when total_purchase_count >= 10 then 2
                when total_purchase_count >= 5 then 8
                when total_purchase_count >= 3 then 15
                when total_purchase_count >= 1 then 25
                else 30
              end
              +
              -- Value factor (0-30 points): Lower LTV = Higher risk
              case
                when total_lifetime_value >= 1000 then 2
                when total_lifetime_value >= 500 then 8
                when total_lifetime_value >= 200 then 15
                when total_lifetime_value >= 100 then 23
                else 30
              end
            )::numeric, 0
          ) as churn_risk_score
        -- Expected next purchase date (based on average purchase interval)
        , case
            when total_purchase_count > 1
                then last_purchase_date + ((last_purchase_date - first_purchase_date) / (total_purchase_count - 1))::int
            else null
          end as expected_next_purchase_date
    from
        customer_purchase_recency
)

, churn_probability as (
    select
        *
        -- Churn probability (0-100%)
        , round((churn_risk_score / 100.0 * 100)::numeric, 2) as churn_probability_pct
        -- Churn risk tier
        , case
            when churn_risk_score >= 75 then 'very_high'
            when churn_risk_score >= 60 then 'high'
            when churn_risk_score >= 40 then 'medium'
            else 'low'
          end as churn_risk_tier
        -- Days until churn threshold (when customer enters "Churned" status based on 180 days)
        , case
            when days_since_last_purchase < 180 then 180 - days_since_last_purchase
            else 0
          end as days_until_churn_threshold
    from
        churn_risk_calculation
)

select
    customer_id
    , first_purchase_date
    , last_purchase_date
    , days_since_last_purchase
    , avg_days_between_purchases
    , expected_next_purchase_date
    , total_purchase_count
    , avg_order_value
    , total_lifetime_value
    , churn_risk_score
    , churn_probability_pct
    , churn_risk_tier
    , days_until_churn_threshold
from
    churn_probability
