-- Seller health scorecard model
-- Grain: seller_id
-- Purpose: Comprehensive seller performance and risk assessment

with fact_order_lines as (
    select
        seller_id
        , order_id
        , product_id
        , item_price
        , item_freight_value
        , review_score
    from {{ ref('fact_order_lines') }}
)

, fact_orders as (
    select
        order_id
        , order_status
        , order_purchase_datetime
        , order_delivered_customer_datetime
        , order_estimated_delivery_datetime
        , payment_total
    from {{ ref('fact_orders') }}
)

, global_metrics as (
    select max(order_purchase_datetime::date) as max_date from fact_orders
)

, seller_orders as (
    select
        fol.seller_id
        , fol.order_id
        , fol.product_id
        , fol.item_price
        , fol.item_freight_value
        , fol.review_score
        , fo.order_status
        , case when fo.order_delivered_customer_datetime <= fo.order_estimated_delivery_datetime then 1 else 0 end as on_time_delivery_flag
        , extract(epoch from (fo.order_delivered_customer_datetime - fo.order_purchase_datetime)) / 3600 as time_to_deliver_hours
        , fo.payment_total
        , fo.order_purchase_datetime
    from
        fact_order_lines fol
    left join fact_orders fo
        on fol.order_id = fo.order_id
)

, seller_metrics as (
    select
        seller_id
        -- Volume metrics
        , count(distinct order_id) as total_orders
        , count(distinct product_id) as unique_products_sold
        , sum(case when order_status not in ('Canceled', 'Unavailable') then 1 else 0 end) as completed_orders
        , sum(case when order_status = 'Canceled' then 1 else 0 end) as canceled_orders
        -- Revenue metrics
        , round(sum(item_price)::numeric, 2) as total_revenue
        , round(sum(payment_total)::numeric, 2) as total_payment
        , round(avg(payment_total)::numeric, 2) as avg_order_value
        -- Delivery performance
        , round(
            (sum(case when on_time_delivery_flag = 1 then 1 else 0 end)::numeric /
             nullif(sum(case when on_time_delivery_flag is not null then 1 else 0 end), 0) * 100)::numeric,
            2
          ) as on_time_delivery_rate
        , round(avg(time_to_deliver_hours)::numeric, 2) as avg_fulfillment_time_hours
        -- Customer satisfaction
        , round(avg(review_score)::numeric, 2) as avg_review_score
        , count(distinct case when review_score >= 4 then order_id end) as satisfied_orders
        , round(
            (count(distinct case when review_score >= 4 then order_id end)::numeric /
             nullif(count(distinct case when review_score is not null then order_id end), 0) * 100)::numeric,
            2
          ) as satisfaction_rate
        -- Tenure metrics
        , min(order_purchase_datetime::date) as first_sale_date
        , max(order_purchase_datetime::date) as last_sale_date
        , (gm.max_date - min(order_purchase_datetime::date)) as seller_tenure_days
    from
        seller_orders
    cross join global_metrics gm
    group by
        seller_id, gm.max_date
)

, health_scoring as (
    select
        seller_id
        , total_orders
        , unique_products_sold
        , completed_orders
        , canceled_orders
        , total_revenue
        , total_payment
        , avg_order_value
        , on_time_delivery_rate
        , avg_fulfillment_time_hours
        , avg_review_score
        , satisfied_orders
        , satisfaction_rate
        , first_sale_date
        , last_sale_date
        , seller_tenure_days
        -- Delivery quality score (0-40)
        , round(
            (
              case
                when on_time_delivery_rate >= 95 then 40
                when on_time_delivery_rate >= 90 then 35
                when on_time_delivery_rate >= 85 then 30
                when on_time_delivery_rate >= 80 then 25
                when on_time_delivery_rate >= 75 then 20
                when on_time_delivery_rate >= 70 then 15
                when on_time_delivery_rate >= 60 then 10
                else 5
              end
            )::numeric,
            0
          ) as delivery_quality_score
        -- Customer satisfaction score (0-35)
        , round(
            (
              case
                when avg_review_score >= 4.7 then 35
                when avg_review_score >= 4.5 then 32
                when avg_review_score >= 4.3 then 28
                when avg_review_score >= 4.0 then 24
                when avg_review_score >= 3.7 then 18
                when avg_review_score >= 3.5 then 12
                else 5
              end
            )::numeric,
            0
          ) as customer_satisfaction_score
        -- Order fulfillment score (0-25)
        , round(
            (
              case
                when completed_orders::numeric / nullif(total_orders, 0) >= 0.95 then 25
                when completed_orders::numeric / nullif(total_orders, 0) >= 0.90 then 22
                when completed_orders::numeric / nullif(total_orders, 0) >= 0.85 then 18
                when completed_orders::numeric / nullif(total_orders, 0) >= 0.80 then 14
                when completed_orders::numeric / nullif(total_orders, 0) >= 0.70 then 10
                else 5
              end
            )::numeric,
            0
          ) as fulfillment_score
    from
        seller_metrics
)

, final_scorecard as (
    select
        seller_id
        , total_orders
        , unique_products_sold
        , completed_orders
        , canceled_orders
        , total_revenue
        , avg_order_value
        , on_time_delivery_rate
        , avg_fulfillment_time_hours
        , avg_review_score
        , satisfaction_rate
        , first_sale_date
        , last_sale_date
        , seller_tenure_days
        , delivery_quality_score
        , customer_satisfaction_score
        , fulfillment_score
        -- Overall performance score (0-100)
        , (delivery_quality_score + customer_satisfaction_score + fulfillment_score) as performance_score
        -- Risk tier
        , case
            when (delivery_quality_score + customer_satisfaction_score + fulfillment_score) >= 85 then 'low'
            when (delivery_quality_score + customer_satisfaction_score + fulfillment_score) >= 65 then 'medium'
            else 'high'
          end as risk_tier
        -- Days to improvement target (days until on_time_delivery_rate reaches 95%)
        , case
            when on_time_delivery_rate < 95
                then ceil((95 - on_time_delivery_rate) / 5)::int
            else 0
          end as days_to_improvement_target
        -- Recommended action
        , case
            when (delivery_quality_score + customer_satisfaction_score + fulfillment_score) >= 85 then 'maintain'
            when (delivery_quality_score + customer_satisfaction_score + fulfillment_score) >= 65 then 'monitor'
            when on_time_delivery_rate < 85 then 'improve_delivery'
            when avg_review_score < 3.5 then 'improve_quality'
            else 'investigate'
          end as recommended_action
    from
        health_scoring
)

select
    seller_id
    , total_orders
    , unique_products_sold
    , completed_orders
    , canceled_orders
    , total_revenue
    , avg_order_value
    , on_time_delivery_rate
    , avg_fulfillment_time_hours
    , avg_review_score
    , satisfaction_rate
    , first_sale_date
    , last_sale_date
    , seller_tenure_days
    , delivery_quality_score
    , customer_satisfaction_score
    , fulfillment_score
    , performance_score
    , risk_tier
    , days_to_improvement_target
    , recommended_action
from
    final_scorecard
