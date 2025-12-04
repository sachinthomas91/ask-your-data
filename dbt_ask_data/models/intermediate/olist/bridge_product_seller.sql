{{
  config(
    materialized='table',
    tags=['bridge', 'olist']
  )
}}

with order_lines as (
  select * from {{ ref('fact_order_lines') }}
)

, orders as (
  select * from {{ ref('fact_orders') }}
)

select
  lin.product_id,
  lin.seller_id,
  min(ord.order_purchase_datetime::date) as first_sale_date,
  max(ord.order_purchase_datetime::date) as last_sale_date,
  count(distinct lin.order_id) as total_orders,
  sum(1) as total_line_items,
  round(sum(lin.item_price)::numeric, 2) as total_revenue,
  round(sum(lin.item_freight_value)::numeric, 2) as total_freight_cost,
  round((sum(lin.item_price) + sum(lin.item_freight_value))::numeric, 2) as total_revenue_with_freight,
  round(avg(lin.item_price)::numeric, 2) as avg_item_price,
  round(avg(lin.review_score)::numeric, 2) as avg_review_score,
  count(distinct lin.order_id) as order_count,
  (current_date - max(ord.order_purchase_datetime::date))::int as days_since_last_sale

from order_lines lin
left join orders ord
  on lin.order_id = ord.order_id
group by
  lin.product_id,
  lin.seller_id
