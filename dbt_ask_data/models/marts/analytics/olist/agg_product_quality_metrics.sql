-- Product quality metrics model
-- Grain: product_id
-- Purpose: Product quality and defect tracking for quality management

with fact_order_lines as (
    select * from {{ ref('fact_order_lines') }}
)

, fact_orders as (
    select * from {{ ref('fact_orders') }}
)

, product_reviews as (
    select
        fol.product_id
        , fol.order_id
        , fol.review_score
        , fol.seller_id
        , ord.order_status
    from
        fact_order_lines fol
    left join fact_orders ord
        on fol.order_id = ord.order_id
)

, quality_metrics as (
    select
        product_id
        -- Review metrics
        , count(distinct case when review_score is not null then order_id end) as total_reviews
        , round(avg(review_score)::numeric, 2) as avg_rating
        -- Review score distribution
        , sum(case when review_score = 5 then 1 else 0 end) as reviews_5_star
        , sum(case when review_score = 4 then 1 else 0 end) as reviews_4_star
        , sum(case when review_score = 3 then 1 else 0 end) as reviews_3_star
        , sum(case when review_score = 2 then 1 else 0 end) as reviews_2_star
        , sum(case when review_score = 1 then 1 else 0 end) as reviews_1_star
        -- Return/Cancellation metrics
        , count(distinct case when order_status = 'Canceled' then order_id end) as return_count
        , round(
            (count(distinct case when order_status = 'Canceled' then order_id end)::numeric /
             nullif(count(distinct order_id), 0) * 100)::numeric,
            2
          ) as return_rate
        -- Satisfaction metrics
        , round(
            (sum(case when review_score >= 4 then 1 else 0 end)::numeric /
             nullif(count(distinct case when review_score is not null then order_id end), 0) * 100)::numeric,
            2
          ) as satisfaction_rate
        -- Total orders
        , count(distinct order_id) as total_orders
    from
        product_reviews
    group by
        product_id
)

, quality_assessment as (
    select
        product_id
        , total_reviews
        , avg_rating
        , reviews_5_star
        , reviews_4_star
        , reviews_3_star
        , reviews_2_star
        , reviews_1_star
        , return_count
        , return_rate
        , satisfaction_rate
        , total_orders
        -- Quality tier based on average rating and satisfaction
        , case
            when avg_rating >= 4.5 and satisfaction_rate >= 90 then 5
            when avg_rating >= 4.0 and satisfaction_rate >= 80 then 4
            when avg_rating >= 3.5 and satisfaction_rate >= 70 then 3
            when avg_rating >= 3.0 and satisfaction_rate >= 60 then 2
            else 1
          end as quality_tier
        -- Defect risk score (0-100)
        , round(
            (
              -- Return rate component (0-50)
              case
                when return_rate <= 5 then 5
                when return_rate <= 10 then 15
                when return_rate <= 15 then 30
                when return_rate <= 20 then 40
                else 50
              end
              +
              -- Rating component (0-50)
              case
                when avg_rating >= 4.5 then 0
                when avg_rating >= 4.0 then 10
                when avg_rating >= 3.5 then 20
                when avg_rating >= 3.0 then 35
                else 50
              end
            )::numeric,
            0
          ) as defect_risk_score
    from
        quality_metrics
)

select
    product_id
    , total_reviews
    , avg_rating
    , reviews_5_star
    , reviews_4_star
    , reviews_3_star
    , reviews_2_star
    , reviews_1_star
    , return_count
    , return_rate
    , satisfaction_rate
    , total_orders
    , quality_tier
    , defect_risk_score
from
    quality_assessment
