with fact_orders as (
    select * from {{ ref('fact_orders') }}
)

, aggregated_daily_sales as (
    select
        order_purchase_datetime::date as order_date
        ,count(order_id) as total_order_count
        ,COUNT(*) FILTER (WHERE order_status = 'Canceled') AS canceled_order_count
        ,COUNT(*) FILTER (WHERE order_status = 'Unavailable') AS unavailable_order_count
        ,round(sum(order_total)::numeric, 2) as total_order_value
        ,round(sum(payment_total)::numeric, 2) as total_paid
        ,round(sum(CASE WHEN order_status = 'Canceled' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_canceled
        ,round(sum(CASE WHEN order_status = 'Unavailable' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_unavailable
        ,round(avg(avg_review_score)::numeric, 2) as avg_review_score
    from
        fact_orders
    group by
        order_purchase_datetime::date
)

select
    *
from
    aggregated_daily_sales