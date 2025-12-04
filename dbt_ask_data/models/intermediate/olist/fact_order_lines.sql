-- Fact table for order line items
-- Grain: order_id, order_item_id (one row per line item in an order)
--
-- Note: Dimensions (dim_products, dim_sellers, dim_customers) are intentionally NOT imported
-- because this is a fact table and should remain in normalized form.
-- Consuming queries should join to dimensions as needed for their specific use case.

with order_lines as (
  select * from {{ ref('stg_order_items')}}
)

,orders as (
  select * from {{ ref('stg_orders')}}
)

,reviews as (
  select * from {{ ref('stg_reviews')}}
)

select
   lin.order_id
  ,lin.order_item_id
  ,ord.order_customer_id
  ,ord.order_status
  ,lin.product_id
  ,lin.seller_id
  ,lin.shipping_limit_datetime
  ,lin.item_price
  ,lin.item_freight_value
  ,rev.review_score
from
  order_lines lin
  left join orders ord
    on lin.order_id=ord.order_id
  left join reviews rev
    on ord.order_id=rev.order_id
