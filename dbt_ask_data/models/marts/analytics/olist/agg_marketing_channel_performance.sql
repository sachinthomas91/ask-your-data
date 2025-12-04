-- Marketing channel performance aggregation
-- Grain: mql_origin_category
-- Purpose: Marketing ROI and lead quality assessment

with stg_mql as (
    select * from {{ ref('stg_marketing_qualified_leads') }}
)

, fact_closed_deals as (
    select * from {{ ref('fact_closed_deals') }}
)

, channel_mql_metrics as (
    select
        mql.mql_origin_category as channel
        , count(distinct mql.mql_id) as mql_count
        , min(mql.mql_first_contact_date) as first_mql_date
        , max(mql.mql_first_contact_date) as last_mql_date
    from
        stg_mql mql
    group by
        mql.mql_origin_category
)

, channel_deal_metrics as (
    select
        cd.mql_origin_category as channel
        , count(distinct cd.mql_id) as deals_closed
        , count(distinct cd.seller_id) as unique_sellers
        , round(sum(cd.lead_business_monthly_revenue)::numeric, 2) as total_declared_revenue
        , round(avg(cd.lead_business_monthly_revenue)::numeric, 2) as avg_deal_value
        , round(avg(cd.days_to_close)::numeric, 2) as avg_days_to_close
    from
        fact_closed_deals cd
    where
        cd.mql_link_status = 'linked'
        and cd.mql_origin_category is not null
    group by
        cd.mql_origin_category
)

, channel_performance as (
    select
        coalesce(cm.channel, dm.channel) as channel
        , coalesce(cm.mql_count, 0) as mql_count
        , coalesce(dm.deals_closed, 0) as deals_closed
        , case
            when coalesce(cm.mql_count, 0) > 0
                then round((coalesce(dm.deals_closed, 0)::numeric / cm.mql_count * 100)::numeric, 2)
            else 0
          end as mql_to_deal_conversion_rate
        , dm.avg_days_to_close
        , dm.avg_deal_value
        , dm.unique_sellers
        , round((coalesce(dm.total_declared_revenue, 0)::numeric / nullif(coalesce(cm.mql_count, 0), 0))::numeric, 2) as seller_ltv_vs_declared_revenue
        -- Channel ROI estimation (deals_closed * avg_deal_value / marketing_cost approximation)
        , round(
            (coalesce(dm.deals_closed, 0)::numeric * coalesce(dm.avg_deal_value, 0)::numeric /
             nullif(coalesce(cm.mql_count, 0), 0))::numeric,
            2
          ) as channel_roi
        -- Quality score (0-100) based on conversion rate and days to close
        , round(
            (
              case
                when (coalesce(dm.deals_closed, 0)::numeric / nullif(coalesce(cm.mql_count, 0), 0) * 100) >= 15 then 40
                when (coalesce(dm.deals_closed, 0)::numeric / nullif(coalesce(cm.mql_count, 0), 0) * 100) >= 10 then 30
                when (coalesce(dm.deals_closed, 0)::numeric / nullif(coalesce(cm.mql_count, 0), 0) * 100) >= 5 then 20
                else 10
              end
              +
              case
                when coalesce(dm.avg_days_to_close, 0) <= 30 then 35
                when coalesce(dm.avg_days_to_close, 0) <= 60 then 25
                when coalesce(dm.avg_days_to_close, 0) <= 90 then 15
                else 5
              end
              +
              case
                when coalesce(dm.avg_deal_value, 0) >= 10000 then 25
                when coalesce(dm.avg_deal_value, 0) >= 5000 then 18
                when coalesce(dm.avg_deal_value, 0) >= 1000 then 10
                else 5
              end
            )::numeric,
            0
          ) as quality_score
    from
        channel_mql_metrics cm
    full outer join channel_deal_metrics dm
        on cm.channel = dm.channel
)

select
    channel
    , mql_count
    , deals_closed
    , mql_to_deal_conversion_rate
    , avg_days_to_close
    , avg_deal_value
    , unique_sellers
    , seller_ltv_vs_declared_revenue
    , channel_roi
    , quality_score
from
    channel_performance
where
    channel is not null
