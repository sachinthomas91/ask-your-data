with order_lines as (
    select * from {{ ref('fact_order_lines') }}
)

, orders as (
    select * from {{ ref('fact_orders') }}
)

, aggregated_daily_sales as (
    select
         lin.seller_id
        ,min(ord.order_purchase_datetime::date) as first_sale_date
        ,max(ord.order_purchase_datetime::date) as last_sale_date
        ,(CURRENT_DATE - max(ord.order_purchase_datetime::date)) as days_since_last_sale
        ,round(sum(ord.order_total)::numeric, 2) as life_time_seller_sales_value
        ,round(avg(ord.order_total)::numeric, 2) as avg_sale_value
        ,count(distinct ord.order_id) as total_order_count
        ,COUNT(*) FILTER (WHERE ord.order_status = 'Canceled') AS canceled_order_count
        ,COUNT(*) FILTER (WHERE ord.order_status = 'Unavailable') AS unavailable_order_count
        ,round(sum(CASE WHEN ord.order_status = 'Canceled' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_canceled
        ,round(sum(CASE WHEN ord.order_status = 'Unavailable' THEN payment_total ELSE 0 END)::numeric, 2) as lost_revenue_unavailable
        ,round(avg(ord.review_score)::numeric, 2) as avg_review_score        
        ,round((count(distinct ord.order_id)::numeric / NULLIF((CURRENT_DATE - min(ord.order_purchase_datetime::date)), 0)) / 30, 2) as avg_monthly_orders
        ,round((sum(ord.order_total)::numeric / NULLIF((CURRENT_DATE - min(ord.order_purchase_datetime::date)), 0)) / 30, 2) as avg_monthly_revenue
        ,round(((COUNT(*) FILTER (WHERE ord.order_status = 'Canceled') + COUNT(*) FILTER (WHERE ord.order_status = 'Unavailable'))::numeric / NULLIF(count(distinct ord.order_id), 0)) * 100, 2) as order_problem_rate
        ,case 
            when (CURRENT_DATE - max(ord.order_purchase_datetime::date)) <= 30 then 'Active'
            when (CURRENT_DATE - max(ord.order_purchase_datetime::date)) <= 90 then 'At Risk'
            when (CURRENT_DATE - max(ord.order_purchase_datetime::date)) <= 180 then 'Churning'
            else 'Churned'
        end as seller_status
    from 
        orders ord
        left join order_lines lin
            on ord.order_id = lin.order_id
    group by 
        lin.seller_id
)

select
    *
from
    aggregated_daily_sales