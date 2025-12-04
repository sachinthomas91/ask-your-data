-- Payments fact table
-- Grain: order_id, payment_type_sequence
-- Purpose: Independent payment analysis without order-level aggregation

with stg_order_payments as (
    select * from {{ ref('stg_order_payments') }}
)

, stg_orders as (
    select * from {{ ref('stg_orders') }}
)

, payments_with_context as (
    select
        pmt.order_id
        , pmt.payment_type_sequence
        , concat(pmt.order_id, '_', pmt.payment_type_sequence) as payment_id
        , pmt.payment_type
        , pmt.payment_value
        , pmt.payment_installment_count
        , ord.order_purchase_datetime as payment_date
        , case
            when ord.order_status in ('Canceled', 'Unavailable') then 1
            else 0
          end as is_refunded
    from
        stg_order_payments pmt
    left join stg_orders ord
        on pmt.order_id = ord.order_id
)

select
    order_id
    , payment_id
    , payment_type_sequence
    , payment_type
    , payment_value
    , payment_installment_count
    , payment_date
    , is_refunded
    , current_timestamp as created_at
from
    payments_with_context
