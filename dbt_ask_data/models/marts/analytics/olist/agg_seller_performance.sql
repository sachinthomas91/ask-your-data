with order_lines as (
    select * from {{ ref('fact_order_lines') }}
)

, orders as (
    select * from {{ ref('fact_orders') }}
)

, global_metrics as (
    select max(order_purchase_datetime::date) as max_date from orders
)

, aggregated_daily_sales as (
    select
         lin.seller_id
        ,min(ord.order_purchase_datetime::date) as first_sale_date
        ,max(ord.order_purchase_datetime::date) as last_sale_date
        ,(gm.max_date - max(ord.order_purchase_datetime::date)) as days_since_last_sale
        ,round(sum(ord.order_total)::numeric, 2) as life_time_seller_sales_value
        ,round(avg(ord.order_total)::numeric, 2) as avg_sale_value
        ,count(distinct ord.order_id) as total_order_count
        ,COUNT(*) FILTER (WHERE ord.order_status = 'Canceled') AS canceled_order_count
        ,COUNT(*) FILTER (WHERE ord.order_status = 'Unavailable') AS unavailable_order_count
        ,round(sum(CASE WHEN ord.order_status = 'Canceled' THEN ord.payment_total ELSE 0 END)::numeric, 2) as lost_revenue_canceled
        ,round(sum(CASE WHEN ord.order_status = 'Unavailable' THEN ord.payment_total ELSE 0 END)::numeric, 2) as lost_revenue_unavailable
        ,round(avg(lin.review_score)::numeric, 2) as avg_review_score
        ,round((count(distinct ord.order_id)::numeric / NULLIF((gm.max_date - min(ord.order_purchase_datetime::date)), 0)) / 30, 2) as avg_monthly_orders
        ,round((sum(ord.order_total)::numeric / NULLIF((gm.max_date - min(ord.order_purchase_datetime::date)), 0)) / 30, 2) as avg_monthly_revenue
        ,round(((COUNT(*) FILTER (WHERE ord.order_status = 'Canceled') + COUNT(*) FILTER (WHERE ord.order_status = 'Unavailable'))::numeric / NULLIF(count(distinct ord.order_id), 0)) * 100, 2) as order_problem_rate
        ,case
            when (gm.max_date - max(ord.order_purchase_datetime::date)) <= 30 then 'Active'
            when (gm.max_date - max(ord.order_purchase_datetime::date)) <= 90 then 'At Risk'
            when (gm.max_date - max(ord.order_purchase_datetime::date)) <= 180 then 'Churning'
            else 'Churned'
        end as seller_status
    from
        order_lines lin
        inner join orders ord
            on lin.order_id = ord.order_id
        cross join global_metrics gm
    group by
        lin.seller_id, gm.max_date
)

select
    *
from
    aggregated_daily_sales