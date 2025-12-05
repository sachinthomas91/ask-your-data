"""
Intelligent Visualization Module for Ask Your Data
===================================================

This module provides context-aware, intelligent data visualization capabilities that:
1. Understand the user's query intent
2. Analyze data characteristics and patterns
3. Automatically select appropriate visualizations
4. Create multi-chart dashboards when beneficial
5. Add statistical insights and annotations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime


class QueryIntent(Enum):
    """Types of analytical intent detected from user queries"""
    COMPARISON = "comparison"  # Compare categories, groups
    TREND = "trend"  # Time-based patterns
    DISTRIBUTION = "distribution"  # Value distribution, frequencies
    RELATIONSHIP = "relationship"  # Correlations, scatter patterns
    COMPOSITION = "composition"  # Part-to-whole, breakdowns
    RANKING = "ranking"  # Top N, bottom N
    AGGREGATION = "aggregation"  # Sums, averages, counts
    UNKNOWN = "unknown"


class ChartType(Enum):
    """Available chart types with business context"""
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    LINE = "line"
    AREA = "area"
    SCATTER = "scatter"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"
    METRIC_CARD = "metric_card"


@dataclass
class ColumnProfile:
    """Statistical profile of a DataFrame column"""
    name: str
    dtype: str
    unique_count: int
    null_count: int
    null_percentage: float
    is_numeric: bool
    is_datetime: bool
    is_categorical: bool
    is_boolean: bool
    is_id: bool
    cardinality: str  # 'low', 'medium', 'high'
    sample_values: List[Any]

    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    # Categorical stats
    top_values: Optional[Dict[Any, int]] = None


@dataclass
class DatasetProfile:
    """Complete profile of a dataset"""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    id_columns: List[str]
    has_time_series: bool
    primary_dimensions: List[str]  # Best categorical grouping columns
    primary_metrics: List[str]  # Best numeric measure columns


class QueryIntentAnalyzer:
    """Analyzes user queries to understand visualization intent"""

    INTENT_PATTERNS = {
        QueryIntent.TREND: [
            r'\b(trend|over time|time series|historical|change|growth|decline)\b',
            r'\b(daily|weekly|monthly|yearly|year|month|day)\b',
            r'\b(progression|evolution|development)\b'
        ],
        QueryIntent.COMPARISON: [
            r'\b(compare|vs|versus|difference|between)\b',
            r'\b(by category|by type|by group|across)\b',
            r'\b(which.*better|which.*more|which.*less)\b'
        ],
        QueryIntent.DISTRIBUTION: [
            r'\b(distribution|spread|range|frequency|histogram)\b',
            r'\b(how many|count of|number of)\b',
            r'\b(breakdown|split)\b'
        ],
        QueryIntent.RANKING: [
            r'\b(top|bottom|best|worst|highest|lowest)\b',
            r'\b(rank|ranking|order by)\b',
            r'\b(most|least)\b'
        ],
        QueryIntent.COMPOSITION: [
            r'\b(composition|makeup|proportion|percentage|share)\b',
            r'\b(part of|portion|segment)\b',
            r'\b(contribute|contribution)\b'
        ],
        QueryIntent.RELATIONSHIP: [
            r'\b(relationship|correlation|connection|related)\b',
            r'\b(impact|affect|influence)\b',
            r'\b(depend|depends on)\b'
        ],
        QueryIntent.AGGREGATION: [
            r'\b(total|sum|average|mean|count|aggregate)\b',
            r'\b(revenue|sales|profit|value)\b'
        ]
    }

    @classmethod
    def analyze_intent(cls, query: str) -> List[QueryIntent]:
        """Analyze query to determine user's analytical intent"""
        query_lower = query.lower()
        detected_intents = []

        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected_intents.append(intent)
                    break

        return detected_intents if detected_intents else [QueryIntent.UNKNOWN]


class DataProfiler:
    """Profiles datasets to understand their characteristics"""

    @staticmethod
    def profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
        """
        Create a detailed statistical profile of a single column.

        Analyzes the column to determine its type (numeric, categorical, ID, etc.),
        cardinality, and key statistics (mean, min, max, top values).

        Args:
            df: The pandas DataFrame containing the column.
            col: The name of the column to profile.

        Returns:
            ColumnProfile object containing all calculated metadata.
        """
        series = df[col]

        # Basic info
        dtype = str(series.dtype)
        unique_count = series.nunique()
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(df)) * 100

        # Type detection
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_boolean = pd.api.types.is_bool_dtype(series) or (unique_count == 2 and is_numeric)

        # ID detection - columns with 'id' in name and high uniqueness
        # Only treat as ID if it has 'id' in name OR if it's a string with high uniqueness
        # Do NOT treat unique numeric columns as IDs automatically (they could be metrics like revenue)
        is_id = False
        if 'id' in col.lower():
             if unique_count > len(df) * 0.8:
                 is_id = True
        elif not is_numeric and unique_count == len(df):
             is_id = True

        # Categorical detection
        is_categorical = False
        if not is_id and not is_datetime:
            if not is_numeric:
                is_categorical = True
            elif is_boolean:
                is_categorical = True
            elif unique_count <= 20:  # Low cardinality numerics
                is_categorical = True

        # Cardinality
        if unique_count <= 10:
            cardinality = 'low'
        elif unique_count <= 50:
            cardinality = 'medium'
        else:
            cardinality = 'high'

        # Sample values
        sample_values = series.dropna().head(5).tolist()

        profile = ColumnProfile(
            name=col,
            dtype=dtype,
            unique_count=unique_count,
            null_count=null_count,
            null_percentage=null_percentage,
            is_numeric=is_numeric and not is_boolean,
            is_datetime=is_datetime,
            is_categorical=is_categorical,
            is_boolean=is_boolean,
            is_id=is_id,
            cardinality=cardinality,
            sample_values=sample_values
        )

        # Numeric statistics
        if is_numeric and not is_boolean:
            profile.mean = float(series.mean())
            profile.median = float(series.median())
            profile.std = float(series.std())
            profile.min_val = float(series.min())
            profile.max_val = float(series.max())

        # Categorical statistics
        if is_categorical:
            top_values = series.value_counts().head(10).to_dict()
            profile.top_values = top_values

        return profile

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
        """
        Create a comprehensive profile of the entire dataset.

        Aggregates individual column profiles and identifies global characteristics
        like primary dimensions (for grouping) and primary metrics (for measuring).

        Args:
            df: The pandas DataFrame to profile.

        Returns:
            DatasetProfile object containing column profiles and dataset-level metadata.
        """
        columns = {col: DataProfiler.profile_column(df, col) for col in df.columns}

        numeric_columns = [col for col, prof in columns.items()
                          if prof.is_numeric and not prof.is_id]
        categorical_columns = [col for col, prof in columns.items()
                              if prof.is_categorical and not prof.is_id]
        datetime_columns = [col for col, prof in columns.items() if prof.is_datetime]
        id_columns = [col for col, prof in columns.items() if prof.is_id]

        # Detect time series
        has_time_series = len(datetime_columns) > 0

        # Identify primary dimensions (best for grouping)
        primary_dimensions = [
            col for col, prof in columns.items()
            if prof.is_categorical and prof.cardinality in ['low', 'medium']
            and not prof.is_id
        ]
        
        # Sort dimensions to prefer non-numeric categorical columns first
        primary_dimensions.sort(key=lambda col: (columns[col].is_numeric, columns[col].unique_count))

        # Identify primary metrics (best for measuring)
        primary_metrics = [
            col for col, prof in columns.items()
            if prof.is_numeric and not prof.is_id
            # Allow metrics with 'id' if they look like counts (e.g. order_id_count)
            # or just include them if they are numeric and not the ID itself
        ]

        return DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            id_columns=id_columns,
            has_time_series=has_time_series,
            primary_dimensions=primary_dimensions,
            primary_metrics=primary_metrics
        )


class IntelligentChartSelector:
    """Selects optimal chart types based on data and query intent"""


    @staticmethod
    def select_charts(
        profile: DatasetProfile,
        intents: List[QueryIntent],
        query: str = ""
    ) -> List[Tuple[ChartType, Dict[str, Any]]]:
        """
        Select appropriate chart types and configurations based on data profile and user intent.

        Uses a "greedy" strategy to recommend multiple complementary visualizations:
        - Trends (Line/Area) if time series data exists.
        - Rankings (Bar) for categorical comparisons.
        - Distributions (Histogram) for numeric metrics.
        - Relationships (Scatter) for multi-metric datasets.

        Args:
            profile: The dataset profile generated by DataProfiler.
            intents: List of detected user intents (e.g., TREND, RANKING).
            query: The original user query string (optional context).

        Returns:
            List of tuples, where each tuple contains:
            - ChartType: The type of chart to generate.
            - Dict: Configuration dictionary for the chart (x, y, title, etc.).
        """
        recommendations = []

        # Priority intent (first detected)
        primary_intent = intents[0] if intents else QueryIntent.UNKNOWN
        
        # 1. TREND ANALYSIS (Time Series)
        # Always check for time series if available, or if explicitly requested
        if profile.has_time_series and (primary_intent == QueryIntent.TREND or True): # "True" makes it greedy
            if profile.primary_metrics:
                # Line chart for primary metric
                recommendations.append((
                    ChartType.LINE,
                    {
                        'x': profile.datetime_columns[0],
                        'y': profile.primary_metrics[0],
                        'title': f'{profile.primary_metrics[0]} Over Time',
                        'color': profile.primary_dimensions[0] if profile.primary_dimensions else None
                    }
                ))

                # Area chart for cumulative metrics if relevant
                if any(word in profile.primary_metrics[0].lower() for word in ['total', 'cumulative', 'sum']):
                    recommendations.append((
                        ChartType.AREA,
                        {
                            'x': profile.datetime_columns[0],
                            'y': profile.primary_metrics[0],
                            'title': f'Cumulative {profile.primary_metrics[0]}'
                        }
                    ))

        # 2. RANKING & COMPARISON (Bar Charts)
        # Always useful if we have categories and metrics
        if profile.primary_dimensions and profile.primary_metrics:
            dim_col = profile.primary_dimensions[0]
            metric_col = profile.primary_metrics[0]
            
            # Avoid self-comparison
            if dim_col == metric_col and len(profile.primary_dimensions) > 1:
                dim_col = profile.primary_dimensions[1]
            
            if dim_col != metric_col:
                # Horizontal bar for ranking (Top N)
                recommendations.append((
                    ChartType.BAR,
                    {
                        'x': metric_col,
                        'y': dim_col,
                        'orientation': 'h',
                        'title': f'Top {dim_col} by {metric_col}',
                        'sort': True
                    }
                ))
                
                # Grouped bar if we have a second dimension
                if len(profile.primary_dimensions) >= 2:
                    recommendations.append((
                        ChartType.GROUPED_BAR,
                        {
                            'x': profile.primary_dimensions[0],
                            'y': profile.primary_metrics[0],
                            'color': profile.primary_dimensions[1],
                            'title': f'{profile.primary_metrics[0]} by {profile.primary_dimensions[0]} and {profile.primary_dimensions[1]}'
                        }
                    ))

        # 3. DISTRIBUTION (Histograms/Box Plots)
        # Useful for understanding data spread
        if profile.primary_metrics:
             recommendations.append((
                ChartType.HISTOGRAM,
                {
                    'x': profile.primary_metrics[0],
                    'title': f'Distribution of {profile.primary_metrics[0]}'
                }
            ))

        # 4. COMPOSITION (Pie/Donut)
        # Only if we have low cardinality dimensions
        if profile.primary_dimensions and profile.primary_metrics:
            dim_profile = profile.columns[profile.primary_dimensions[0]]
            if dim_profile.cardinality == 'low':
                recommendations.append((
                    ChartType.PIE,
                    {
                        'names': profile.primary_dimensions[0],
                        'values': profile.primary_metrics[0],
                        'title': f'{profile.primary_metrics[0]} Distribution by {profile.primary_dimensions[0]}'
                    }
                ))

        # 5. RELATIONSHIP (Scatter)
        # If we have multiple metrics
        if len(profile.primary_metrics) >= 2:
             recommendations.append((
                ChartType.SCATTER,
                {
                    'x': profile.primary_metrics[0],
                    'y': profile.primary_metrics[1],
                    'color': profile.primary_dimensions[0] if profile.primary_dimensions else None,
                    'title': f'{profile.primary_metrics[1]} vs {profile.primary_metrics[0]}'
                }
            ))

        # Deduplicate recommendations based on title and type
        unique_recommendations = []
        seen_configs = set()
        
        for chart_type, config in recommendations:
            config_key = (chart_type, config.get('title'))
            if config_key not in seen_configs:
                unique_recommendations.append((chart_type, config))
                seen_configs.add(config_key)

        # Fallback: If no charts yet, try to use ID columns for Top N
        if not unique_recommendations and profile.id_columns and profile.primary_metrics:
            # Use the first ID column as a dimension
            id_col = profile.id_columns[0]
            metric_col = profile.primary_metrics[0]
            
            unique_recommendations.append((
                ChartType.BAR,
                {
                    'x': metric_col,
                    'y': id_col,
                    'orientation': 'h',
                    'title': f'Top {id_col} by {metric_col}',
                    'sort': True
                }
            ))

        # Limit to reasonable number of charts (e.g., 4) to avoid overwhelming
        return unique_recommendations[:4]

    @staticmethod
    def _get_fallback_charts(profile: DatasetProfile) -> List[Tuple[ChartType, Dict[str, Any]]]:
        """Fallback chart recommendations when intent is unclear"""
        recommendations = []

        # If we have dimensions and metrics, create bar chart
        if profile.primary_dimensions and profile.primary_metrics:
            recommendations.append((
                ChartType.BAR,
                {
                    'x': profile.primary_dimensions[0],
                    'y': profile.primary_metrics[0],
                    'title': f'{profile.primary_metrics[0]} by {profile.primary_dimensions[0]}'
                }
            ))

        # If we have time series, create line chart
        elif profile.datetime_columns and profile.primary_metrics:
            recommendations.append((
                ChartType.LINE,
                {
                    'x': profile.datetime_columns[0],
                    'y': profile.primary_metrics[0],
                    'title': f'{profile.primary_metrics[0]} Over Time'
                }
            ))

        # If we have multiple metrics (3+), create multi-metric visualizations
        elif len(profile.primary_metrics) >= 3:
            # Scatter plot for relationship analysis
            recommendations.append((
                ChartType.SCATTER,
                {
                    'x': profile.primary_metrics[0],
                    'y': profile.primary_metrics[1],
                    'color': profile.primary_metrics[2] if len(profile.primary_metrics) > 2 else None,
                    'title': f'{profile.primary_metrics[1]} vs {profile.primary_metrics[0]} (colored by {profile.primary_metrics[2]})'
                }
            ))

            # Heatmap for correlation analysis
            recommendations.append((
                ChartType.HEATMAP,
                {
                    'title': 'Correlation Matrix - All Numeric Columns'
                }
            ))

        # If we have exactly 2 metrics, create scatter plot
        elif len(profile.primary_metrics) == 2:
            recommendations.append((
                ChartType.SCATTER,
                {
                    'x': profile.primary_metrics[0],
                    'y': profile.primary_metrics[1],
                    'title': f'{profile.primary_metrics[1]} vs {profile.primary_metrics[0]}'
                }
            ))

        # Single metric - histogram + box plot
        elif profile.primary_metrics:
            recommendations.append((
                ChartType.HISTOGRAM,
                {
                    'x': profile.primary_metrics[0],
                    'title': f'Distribution of {profile.primary_metrics[0]}'
                }
            ))

            recommendations.append((
                ChartType.BOX,
                {
                    'y': profile.primary_metrics[0],
                    'title': f'Statistical Summary of {profile.primary_metrics[0]}'
                }
            ))

        return recommendations


class SmartAggregator:
    """Intelligently aggregates data for meaningful visualizations"""


    @staticmethod
    def aggregate_for_chart(
        df: pd.DataFrame,
        chart_type: ChartType,
        config: Dict[str, Any],
        profile: DatasetProfile
    ) -> pd.DataFrame:
        """
        Apply smart aggregation to the DataFrame based on the chart type.

        Handles:
        - Sorting and limiting for "Top N" bar charts.
        - Grouping and summing/counting for categorical charts.
        - Time-series sorting for line charts.
        - Collision prevention (e.g., when grouping by a column that is also the metric).

        Args:
            df: The raw pandas DataFrame.
            chart_type: The type of chart being generated.
            config: The chart configuration dictionary.
            profile: The dataset profile.

        Returns:
            A new, aggregated pandas DataFrame ready for plotting.
        """

        # For ranking charts, sort and limit
        if chart_type == ChartType.BAR and config.get('sort'):
            dim_col = config.get('y') if config.get('orientation') == 'h' else config.get('x')
            metric_col = config.get('x') if config.get('orientation') == 'h' else config.get('y')

            if dim_col and metric_col:
                # Handle case where dimension and metric are the same column
                if dim_col == metric_col:
                    # If they are the same, we can't group by it and sum it meaningfully for a ranking
                    # Instead, we should just take the top values
                    agg_df = df.nlargest(10, metric_col)
                else:
                    # Group and aggregate
                    agg_df = df.groupby(dim_col)[metric_col].sum().reset_index()
                    # Sort and take top 10
                    agg_df = agg_df.nlargest(10, metric_col)
                
                return agg_df

        # For categorical groupings, aggregate if needed
        if chart_type in [ChartType.BAR, ChartType.GROUPED_BAR, ChartType.PIE]:
            x_col = config.get('x') or config.get('names')
            y_col = config.get('y') or config.get('values')
            color_col = config.get('color')

            if x_col and y_col:
                # Check if aggregation is needed
                if profile.columns[x_col].is_categorical:
                    group_cols = [x_col]
                    if color_col and profile.columns[color_col].is_categorical:
                        group_cols.append(color_col)

                    # Aggregate - sum if looks like a metric, otherwise count
                    if any(word in y_col.lower() for word in ['amount', 'total', 'sum', 'revenue', 'price', 'value']):
                        # Handle collision if grouping by the same column we are summing
                        if y_col in group_cols:
                            # If we are grouping by the value, we probably want a count of occurrences, not a sum of the value itself
                            # Or we just return the unique values
                            agg_df = df.groupby(group_cols).size().reset_index(name='count')
                            # Swap y_col to be the count
                            # Note: This changes the chart semantics, but prevents a crash
                        else:
                            agg_df = df.groupby(group_cols)[y_col].sum().reset_index()
                    else:
                        if y_col in group_cols:
                             agg_df = df.groupby(group_cols).size().reset_index(name='count')
                        else:
                            agg_df = df.groupby(group_cols)[y_col].mean().reset_index()

                    return agg_df

        # For time series, ensure proper sorting
        if chart_type in [ChartType.LINE, ChartType.AREA]:
            x_col = config.get('x')
            if x_col and profile.columns[x_col].is_datetime:
                return df.sort_values(x_col)

        return df


class VisualizationEngine:
    """Main engine for creating intelligent visualizations"""


    @staticmethod
    def create_chart(
        df: pd.DataFrame,
        chart_type: ChartType,
        config: Dict[str, Any],
        profile: DatasetProfile
    ) -> Optional[go.Figure]:
        """
        Create a Plotly figure based on the specified chart type and configuration.

        Orchestrates the visualization process:
        1. Validates required columns.
        2. Aggregates data using SmartAggregator.
        3. Generates the specific Plotly figure.
        4. Adds statistical insights/annotations.
        5. Applies standard layout styling.

        Args:
            df: The raw pandas DataFrame.
            chart_type: The type of chart to create.
            config: Configuration dictionary (must contain keys like 'x', 'y', etc.).
            profile: The dataset profile.

        Returns:
            A plotly.graph_objects.Figure object, or None if creation fails.
        """

        try:
            # Validate required columns exist in dataframe
            required_cols = []
            if config.get('x'):
                required_cols.append(config['x'])
            if config.get('y'):
                required_cols.append(config['y'])
            if config.get('names'):
                required_cols.append(config['names'])
            if config.get('values'):
                required_cols.append(config['values'])
            if config.get('color'):
                required_cols.append(config['color'])

            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in dataframe for {chart_type.value} chart")
                    return None

            # Apply smart aggregation
            plot_df = SmartAggregator.aggregate_for_chart(df, chart_type, config, profile)

            if plot_df.empty:
                print(f"Warning: No data after aggregation for {chart_type.value} chart")
                return None

            # Create chart based on type
            if chart_type == ChartType.BAR:
                fig = VisualizationEngine._create_bar_chart(plot_df, config)
            elif chart_type == ChartType.GROUPED_BAR:
                fig = VisualizationEngine._create_grouped_bar(plot_df, config)
            elif chart_type == ChartType.LINE:
                fig = VisualizationEngine._create_line_chart(plot_df, config)
            elif chart_type == ChartType.AREA:
                fig = VisualizationEngine._create_area_chart(plot_df, config)
            elif chart_type == ChartType.SCATTER:
                fig = VisualizationEngine._create_scatter(plot_df, config)
            elif chart_type == ChartType.PIE:
                fig = VisualizationEngine._create_pie_chart(plot_df, config)
            elif chart_type == ChartType.HISTOGRAM:
                fig = VisualizationEngine._create_histogram(plot_df, config)
            elif chart_type == ChartType.BOX:
                fig = VisualizationEngine._create_box_plot(plot_df, config)
            elif chart_type == ChartType.HEATMAP:
                fig = VisualizationEngine._create_heatmap(plot_df, config)
            else:
                return None

            if fig is None:
                print(f"Warning: Failed to create {chart_type.value} chart")
                return None

            # Add statistical annotations
            VisualizationEngine._add_insights(fig, plot_df, chart_type, config, profile)

            # Standard layout improvements
            fig.update_layout(
                height=500,
                showlegend=True,
                template='plotly_white',
                hovermode='closest'
            )

            return fig

        except Exception as e:
            print(f"Error creating {chart_type.value} chart: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _create_bar_chart(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        orientation = config.get('orientation', 'v')

        if orientation == 'h':
            fig = px.bar(df, x=config['x'], y=config['y'],
                        orientation='h', title=config.get('title', ''))
        else:
            fig = px.bar(df, x=config['x'], y=config['y'],
                        title=config.get('title', ''))

        return fig

    @staticmethod
    def _create_grouped_bar(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create grouped bar chart"""
        fig = px.bar(df, x=config['x'], y=config['y'], color=config.get('color'),
                    barmode='group', title=config.get('title', ''))
        return fig

    @staticmethod
    def _create_line_chart(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        fig = px.line(df, x=config['x'], y=config['y'],
                     color=config.get('color'),
                     title=config.get('title', ''))

        # Add markers for better visibility
        fig.update_traces(mode='lines+markers')
        return fig

    @staticmethod
    def _create_area_chart(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create area chart"""
        fig = px.area(df, x=config['x'], y=config['y'],
                     title=config.get('title', ''))
        return fig

    @staticmethod
    def _create_scatter(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        color_col = config.get('color')

        # Only use color if it exists in dataframe
        if color_col and color_col not in df.columns:
            color_col = None

        # Add trendline if appropriate
        if len(df) > 10:
            fig = px.scatter(df, x=config['x'], y=config['y'],
                           color=color_col,
                           # trendline='ols', # Requires statsmodels, removing for stability
                           title=config.get('title', ''))
        else:
            fig = px.scatter(df, x=config['x'], y=config['y'],
                           color=color_col,
                           title=config.get('title', ''))

        return fig

    @staticmethod
    def _create_pie_chart(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        # Limit to top categories if too many
        if len(df) > 8:
            metric_col = config['values']
            df = df.nlargest(8, metric_col)

        fig = px.pie(df, names=config['names'], values=config['values'],
                    title=config.get('title', ''))

        # Improve readability
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @staticmethod
    def _create_histogram(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        fig = px.histogram(df, x=config['x'], title=config.get('title', ''))
        return fig

    @staticmethod
    def _create_box_plot(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        fig = px.box(df, x=config.get('x'), y=config['y'],
                    title=config.get('title', ''))
        return fig

    @staticmethod
    def _create_heatmap(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap"""
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig = px.imshow(corr, text_auto=True, aspect='auto',
                       title=config.get('title', 'Correlation Heatmap'))
        return fig

    @staticmethod
    def _add_insights(
        fig: go.Figure,
        df: pd.DataFrame,
        chart_type: ChartType,
        config: Dict[str, Any],
        profile: DatasetProfile
    ):
        """Add statistical insights and annotations to charts"""

        # Add mean line to histograms
        if chart_type == ChartType.HISTOGRAM:
            x_col = config['x']
            if x_col in df.columns:
                mean_val = df[x_col].mean()
                fig.add_vline(x=mean_val, line_dash='dash', line_color='red',
                             annotation_text=f'Mean: {mean_val:.2f}')

        # Add trend annotations to line charts
        elif chart_type == ChartType.LINE:
            y_col = config['y']
            if y_col in df.columns and len(df) > 1:
                # Calculate trend
                first_val = df[y_col].iloc[0]
                last_val = df[y_col].iloc[-1]
                pct_change = ((last_val - first_val) / first_val) * 100

                trend_text = f"{'↑' if pct_change > 0 else '↓'} {abs(pct_change):.1f}% overall"
                fig.add_annotation(
                    text=trend_text,
                    xref="paper", yref="paper",
                    x=0.95, y=0.95,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray"
                )


class IntelligentVisualizer:
    """Main class for intelligent, context-aware visualization"""

    def __init__(self):
        self.intent_analyzer = QueryIntentAnalyzer()
        self.profiler = DataProfiler()
        self.selector = IntelligentChartSelector()
        self.engine = VisualizationEngine()

    def create_visualizations(
        self,
        df: pd.DataFrame,
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Create intelligent visualizations based on data and query.

        Returns:
            Dictionary containing:
            - profile: DatasetProfile
            - intents: List of detected intents
            - charts: List of created Plotly figures
            - insights: Textual insights about the data
        """

        if df.empty:
            return {
                'profile': None,
                'intents': [],
                'charts': [],
                'insights': "No data available for visualization."
            }

        # Step 1: Profile the data
        profile = self.profiler.profile_dataset(df)
        print(f"[DEBUG] Dataset profile: {profile.row_count} rows, {profile.column_count} columns")
        print(f"[DEBUG] Primary metrics: {profile.primary_metrics}")
        print(f"[DEBUG] Primary dimensions: {profile.primary_dimensions}")
        print(f"[DEBUG] Numeric columns: {profile.numeric_columns}")
        print(f"[DEBUG] Categorical columns: {profile.categorical_columns}")

        # Step 2: Analyze query intent
        intents = self.intent_analyzer.analyze_intent(query)
        print(f"[DEBUG] Detected intents: {[i.value for i in intents]}")

        # Step 3: Select appropriate charts
        chart_recommendations = self.selector.select_charts(profile, intents, query)
        print(f"[DEBUG] Chart recommendations: {[(c.value, cfg.get('title', '')) for c, cfg in chart_recommendations]}")

        # Step 4: Create visualizations
        charts = []
        for chart_type, config in chart_recommendations:
            print(f"[DEBUG] Creating {chart_type.value} chart with config: {config}")
            fig = self.engine.create_chart(df, chart_type, config, profile)
            if fig:
                print(f"[DEBUG] Successfully created {chart_type.value} chart")
                charts.append({
                    'type': chart_type.value,
                    'figure': fig,
                    'title': config.get('title', ''),
                    'config': config
                })
            else:
                print(f"[DEBUG] Failed to create {chart_type.value} chart")

        # Step 5: Generate insights
        insights = self._generate_insights(df, profile, intents, chart_recommendations, charts)

        return {
            'profile': profile,
            'intents': [i.value for i in intents],
            'charts': charts,
            'insights': insights
        }

    def _generate_insights(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        intents: List[QueryIntent],
        chart_recommendations: List[Tuple[ChartType, Dict[str, Any]]] = None,
        charts: List[Dict[str, Any]] = None
    ) -> str:
        """Generate textual insights about the data"""

        insights = []

        # Dataset overview
        insights.append(f"**Dataset Overview**: {profile.row_count:,} rows, {profile.column_count} columns")

        # Key metrics
        if profile.primary_metrics:
            insights.append(f"**Key Metrics**: {', '.join(profile.primary_metrics[:3])}")

            # Statistics for primary metric
            primary_metric = profile.primary_metrics[0]
            col_profile = profile.columns[primary_metric]
            if col_profile.mean is not None:
                insights.append(
                    f"**{primary_metric}**: "
                    f"Avg: {col_profile.mean:,.2f}, "
                    f"Range: {col_profile.min_val:,.2f} - {col_profile.max_val:,.2f}"
                )

        # Key dimensions
        if profile.primary_dimensions:
            insights.append(f"**Dimensions**: {', '.join(profile.primary_dimensions[:3])}")

        # Time series info
        if profile.has_time_series:
            insights.append(f"**Time Period**: Data includes time-series information")

        # Intent-specific insights
        primary_intent = intents[0] if intents else None
        if primary_intent == QueryIntent.RANKING and profile.primary_metrics:
            insights.append("**Analysis Type**: Ranking/Top performers")
        elif primary_intent == QueryIntent.TREND:
            insights.append("**Analysis Type**: Trend analysis over time")
        elif primary_intent == QueryIntent.COMPARISON:
            insights.append("**Analysis Type**: Comparative analysis")

        # Helpful guidance if visualizations couldn't be created
        if chart_recommendations and not charts:
            insights.append(
                "**Visualization Note**: The data structure (primarily numeric metrics without categorical grouping) "
                "makes standard comparisons difficult. Try refining your query to include categorical dimensions like "
                "dates, categories, regions, statuses, or business segments to enable more meaningful visualizations."
            )

        return "\n\n".join(insights)
