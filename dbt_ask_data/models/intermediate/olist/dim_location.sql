-- Location dimension
-- Grain: zip_code
-- Purpose: Geographic hierarchy and location enrichment for customer and seller analysis

with stg_geolocation as (
    select * from {{ ref('stg_geolocation') }}
)

, unique_locations as (
    select distinct
        zip_code
        , city
        , state
    from
        stg_geolocation
)

, regional_mapping as (
    select
        zip_code
        , city
        , state
        -- Map states to regions
        , case
            when state in ('RJ', 'SP', 'ES', 'MG') then 'Southeast'
            when state in ('PR', 'SC', 'RS') then 'South'
            when state in ('SP', 'MG', 'GO', 'DF') then 'Central-West'
            when state in ('AM', 'RR', 'AP', 'PA', 'TO', 'AC') then 'North'
            when state in ('CE', 'RN', 'PE', 'AL', 'SE', 'BA', 'PI', 'MA') then 'Northeast'
            else 'Unknown'
          end as region
        -- Metropolitan area classification (based on major cities)
        , case
            when lower(city) in ('são paulo', 'sp') then 'São Paulo Metro'
            when lower(city) in ('rio de janeiro', 'rj') then 'Rio de Janeiro Metro'
            when lower(city) in ('belo horizonte', 'mg') then 'Belo Horizonte Metro'
            when lower(city) in ('brasília', 'df') then 'Brasília'
            when lower(city) in ('salvador', 'ba') then 'Salvador Metro'
            when lower(city) in ('fortaleza', 'ce') then 'Fortaleza Metro'
            when lower(city) in ('recife', 'pe') then 'Recife Metro'
            when lower(city) in ('porto alegre', 'rs') then 'Porto Alegre Metro'
            when lower(city) in ('curitiba', 'pr') then 'Curitiba Metro'
            when lower(city) in ('manaus', 'am') then 'Manaus Metro'
            else 'Non-Metro'
          end as metro_area
        -- Generate geographic_region_id for hierarchy
        , concat(state, '_', row_number() over (partition by state order by city)) as geographic_region_id
    from
        unique_locations
)

select
    zip_code
    , city
    , state
    , region
    , metro_area
    , geographic_region_id
    , current_timestamp as created_at
from
    regional_mapping
