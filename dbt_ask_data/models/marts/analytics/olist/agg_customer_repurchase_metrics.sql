-- Customer repurchase metrics model
-- Grain: customer_id
-- Purpose: Analyze repeat purchase behavior and customer lifetime value

with customer_purchases as (
    select
        customer_id
        , count(distinct order_id) as total_purchases
        , min(order_purchase_datetime::date) as first_purchase_date
        , max(order_purchase_datetime::date) as last_purchase_date
        , round(sum(order_total)::numeric, 2) as repeat_customer_lifetime_value
        , round(avg(order_total)::numeric, 2) as avg_repeat_order_value
    from {{ ref('fact_orders') }}
    where
        order_status not in ('Canceled', 'Unavailable')
    group by
        customer_id
)

, repeat_metrics as (
    select
        customer_id
        , total_purchases
        , first_purchase_date
        , last_purchase_date
        , repeat_customer_lifetime_value
        , avg_repeat_order_value
        -- Repeat customer indicator
        , case when total_purchases > 1 then 1 else 0 end as is_repeat_customer
        -- Repeat purchase rate
        , case
            when total_purchases > 1
                then round(((total_purchases - 1)::numeric / total_purchases::numeric * 100), 2)
            else 0
          end as repeat_purchase_rate
        -- Average days between purchases (simple average)
        , case
            when total_purchases > 1
                then round(((last_purchase_date - first_purchase_date)::numeric / (total_purchases - 1)), 2)
            else null
          end as avg_days_between_purchases
        -- Days until first repeat purchase (approximated as 1/2 of average interval)
        , case
            when total_purchases > 1
                then ceil(((last_purchase_date - first_purchase_date)::numeric / (total_purchases - 1) / 2))::int
            else null
          end as first_repeat_purchase_days
    from
        customer_purchases
)

, repurchase_probability as (
    select
        customer_id
        , total_purchases
        , is_repeat_customer
        , repeat_purchase_rate
        , avg_days_between_purchases
        , first_repeat_purchase_days
        , repeat_customer_lifetime_value
        , avg_repeat_order_value
        , last_purchase_date
        -- Purchase count tier
        , case
            when total_purchases >= 10 then 'high_frequency'
            when total_purchases >= 5 then 'medium_frequency'
            when total_purchases >= 3 then 'occasional'
            else 'one_time'
          end as purchase_count_tier
        -- Repurchase probability (0-100%)
        , round(
            (
              case
                when is_repeat_customer = 0 then 10
                when total_purchases >= 10 then 95
                when total_purchases >= 5 then 80
                when total_purchases >= 3 then 60
                else 30
              end
            )::numeric,
            2
          ) as repurchase_probability_pct
    from
        repeat_metrics
)

select
    customer_id
    , total_purchases
    , is_repeat_customer
    , repeat_purchase_rate
    , avg_days_between_purchases
    , first_repeat_purchase_days
    , repeat_customer_lifetime_value
    , avg_repeat_order_value
    , last_purchase_date
    , purchase_count_tier
    , repurchase_probability_pct
from
    repurchase_probability
