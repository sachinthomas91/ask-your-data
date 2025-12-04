-- Reviews fact table
-- Grain: review_id
-- Purpose: Independent review analysis without order-level dependencies

with stg_reviews as (
    select * from {{ ref('stg_reviews') }}
)

, stg_orders as (
    select * from {{ ref('stg_orders') }}
)

, stg_order_items as (
    select * from {{ ref('stg_order_items') }}
)

, reviews_with_context as (
    select
        rev.review_id
        , rev.order_id
        , ord.order_customer_id as customer_id
        , rev.review_score
        , rev.review_comment_title
        , rev.review_comment_message
        , coalesce(length(rev.review_comment_message), 0) as review_comment_length
        , case when rev.review_comment_message is not null then 1 else 0 end as has_comment_flag
        , rev.review_creation_datetime as review_date
        , rev.review_answer_datetime
        , ord.order_purchase_datetime
        , extract(day from (rev.review_creation_datetime - ord.order_purchase_datetime)) as days_to_review
    from
        stg_reviews rev
    left join stg_orders ord
        on rev.order_id = ord.order_id
)

select
    review_id
    , order_id
    , customer_id
    , review_score
    , review_comment_title
    , review_comment_message
    , review_comment_length
    , has_comment_flag
    , review_date
    , review_answer_datetime
    , order_purchase_datetime
    , days_to_review
    , current_timestamp as created_at
from
    reviews_with_context
