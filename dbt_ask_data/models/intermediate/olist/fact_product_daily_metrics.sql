-- Product daily metrics fact table
-- Grain: product_id, order_date
-- Purpose: Daily-level product performance metrics for trend analysis

with fact_order_lines as (
    select * from {{ ref('fact_order_lines') }}
)

, stg_products as (
    select * from {{ ref('stg_products') }}
)

, joined_data as (
    select
        fol.product_id
        , fol.order_id
        , ord.order_purchase_datetime::date as order_date
        , fol.item_price
        , fol.item_freight_value
        , fol.review_score
        , ord.order_status
        , ord.customer_id
        , fol.seller_id
    from
        fact_order_lines fol
    left join {{ ref('fact_orders') }} ord
        on fol.order_id = ord.order_id
)

, daily_product_metrics as (
    select
        product_id
        , order_date
        -- Unit metrics
        , count(distinct order_id) as units_sold
        , sum(case when order_status = 'Canceled' then 1 else 0 end) as units_refunded
        -- Revenue metrics
        , round(sum(item_price)::numeric, 2) as gross_revenue
        , round(sum(item_freight_value)::numeric, 2) as freight_revenue
        , round((sum(item_price) + sum(item_freight_value))::numeric, 2) as total_revenue
        -- Customer metrics
        , count(distinct customer_id) as distinct_buyers
        , count(distinct seller_id) as distinct_sellers
        -- Review metrics
        , round(avg(review_score)::numeric, 2) as avg_review_score
        , count(distinct case when review_score is not null then order_id end) as review_count
        -- Price metrics
        , round(min(item_price)::numeric, 2) as price_min
        , round(avg(item_price)::numeric, 2) as price_avg
        , round(max(item_price)::numeric, 2) as price_max
        -- Freight cost percentage
        , round(
            (sum(item_freight_value) / nullif(sum(item_price), 0) * 100)::numeric,
            2
          ) as freight_cost_pct_of_revenue
    from
        joined_data
    where
        order_status not in ('Canceled', 'Unavailable')
    group by
        product_id
        , order_date
)

select
    product_id
    , order_date
    , units_sold
    , units_refunded
    , gross_revenue
    , freight_revenue
    , total_revenue
    , distinct_buyers
    , distinct_sellers
    , avg_review_score
    , review_count
    , price_min
    , price_avg
    , price_max
    , freight_cost_pct_of_revenue
    , current_timestamp as created_at
from
    daily_product_metrics
