with fact_orders as (
    select * from {{ ref('fact_orders') }}
)

, aggregated_daily_sales as (
    select
        customer_id
        ,min(order_purchase_datetime::date) as first_purchase_date
        ,max(order_purchase_datetime::date) as last_purchase_date
        ,(CURRENT_DATE - max(order_purchase_datetime::date)) as days_since_last_purchase
        ,(max(order_purchase_datetime::date) - min(order_purchase_datetime::date)) as customer_lifetime_days
        ,round(sum(order_total)::numeric, 2) as life_time_customer_value
        ,round(avg(order_total)::numeric, 2) as avg_order_value
        ,count(order_id) as total_order_count
        ,COUNT(*) FILTER (WHERE order_status = 'Canceled') AS canceled_order_count
        ,COUNT(*) FILTER (WHERE order_status = 'Unavailable') AS unavailable_order_count
        ,round(sum(CASE WHEN order_status = 'Canceled' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_canceled
        ,round(sum(CASE WHEN order_status = 'Unavailable' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_unavailable
        ,round(avg(review_score)::numeric, 2) as avg_review_score
        -- Derived Metrics
        ,round((count(order_id)::numeric / NULLIF((max(order_purchase_datetime::date) - min(order_purchase_datetime::date)), 0)) * 30, 2) as avg_monthly_orders
        ,round((sum(order_total)::numeric / NULLIF((max(order_purchase_datetime::date) - min(order_purchase_datetime::date)), 0)) * 30, 2) as avg_monthly_spend
        ,round((COUNT(*) FILTER (WHERE order_status IN ('Canceled', 'Unavailable'))::numeric / NULLIF(count(order_id), 0)) * 100, 2) as problem_order_rate
        ,round(stddev_samp(order_total)::numeric, 2) as order_value_stddev
        ,round((SUM(CASE WHEN review_score >= 4 THEN 1 ELSE 0 END)::numeric / NULLIF(count(review_score), 0)) * 100, 2) as positive_review_rate
        -- Customer Status
        ,case 
            when (CURRENT_DATE - max(order_purchase_datetime::date)) <= 30 then 'Active'
            when (CURRENT_DATE - max(order_purchase_datetime::date)) <= 90 then 'At Risk'
            when (CURRENT_DATE - max(order_purchase_datetime::date)) <= 180 then 'Churning'
            else 'Churned'
        end as customer_status
        -- Customer Value Tier based on avg monthly spend
        ,case 
            when round((sum(order_total)::numeric / NULLIF((max(order_purchase_datetime::date) - min(order_purchase_datetime::date)), 0)) * 30, 2) >= 1000 then 'High Value'
            when round((sum(order_total)::numeric / NULLIF((max(order_purchase_datetime::date) - min(order_purchase_datetime::date)), 0)) * 30, 2) >= 500 then 'Medium Value'
            else 'Low Value'
        end as customer_value_tier
    from
        fact_orders
    group by 
        customer_id
)

select
    *
from
    aggregated_daily_sales