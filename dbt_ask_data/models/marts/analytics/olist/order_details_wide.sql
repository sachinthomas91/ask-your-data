with order_lines as (
    select * from {{ ref('fact_order_lines') }}
)

, orders as (
    select * from {{ ref('fact_orders') }}
)

select
    -- Order Line Details
    lin.order_id,
    lin.order_item_id,
    lin.shipping_limit_datetime,
    lin.item_price,
    lin.item_freight_value,
    lin.review_score,
    lin.product_id,
    lin.seller_id,
    lin.order_customer_id as customer_id,

    -- Order Details
    ord.order_status,
    ord.order_item_count,
    ord.order_item_price_total,
    ord.order_freight_total,
    ord.order_total,
    ord.payment_count,
    ord.payment_method_type_count,
    ord.payment_installment_count,
    ord.payment_total,
    ord.avg_review_score,
    ord.time_to_approval_hours,
    ord.time_to_ship_hours,
    ord.time_to_deliver_hours,
    ord.delivery_estimation_slip_hours,
    ord.order_purchase_datetime,
    ord.order_approved_datetime,
    ord.order_delivered_carrier_datetime,
    ord.order_delivered_customer_datetime,
    ord.order_estimated_delivery_datetime
from
    order_lines lin
    left join orders ord
        on lin.order_id = ord.order_id

