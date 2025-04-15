import os
import re
import tempfile
import json
import random

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyecharts import options as opts
from pyecharts.charts import Sunburst
from pyecharts.globals import ThemeType


# Function to generate publications by year chart data
def get_publications_by_year(df, matching_docs):
    """
    Create a DataFrame with publication counts by year
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    
    Returns:
    pandas.DataFrame: DataFrame with Year and Count columns
    """
    year_counts = {}
    
    if 'publication_year' in df['Category'].values:
        year_rows = df[df['Category'] == 'publication_year']
        
        for doc_col in matching_docs:
            year_value = year_rows[doc_col].iloc[0] if not year_rows.empty else None
            if year_value and not pd.isna(year_value):
                try:
                    year = int(float(year_value))
                    if year in year_counts:
                        year_counts[year] += 1
                    else:
                        year_counts[year] = 1
                except (ValueError, TypeError):
                    continue
    
    # Convert to DataFrame
    if year_counts:
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        return pd.DataFrame({'Year': years, 'Count': counts})
    
    return pd.DataFrame()

def generate_pyecharts_sunburst_data(df, matching_docs):
    """
    Generate hierarchical data structure for pyecharts sunburst chart
    from filtered matching_docs, showing top 5 from each hierarchy level:
    publication type, study design, and funding source
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    
    Returns:
    list: Nested dictionary structure for the sunburst chart
    """
    # Extract data from filtered documents
    pub_types = {}
    study_designs = {}
    funding_sources = {}
    
    # Get rows for each category
    pub_type_rows = df[df['Category'] == 'publication_type']
    study_design_rows = df[df['SubCategory'] == 'primary_type']
    funding_rows = df[df['SubCategory'] == 'type']
    
    # Track document relationships between categories
    relationships = {}
    
    # Count occurrences and track relationships
    for doc_col in matching_docs:
        # Extract publication type
        if not pub_type_rows.empty:
            try:
                pub_type = pub_type_rows[doc_col].iloc[0]
                if pub_type and not pd.isna(pub_type):
                    pub_types[pub_type] = pub_types.get(pub_type, 0) + 1
                    
                    # Extract study design for this document
                    if not study_design_rows.empty:
                        try:
                            design = study_design_rows[doc_col].iloc[0]
                            if design and not pd.isna(design):
                                study_designs[design] = study_designs.get(design, 0) + 1
                                
                                # Create relationship key
                                rel_key = f"{pub_type}|{design}"
                                if rel_key not in relationships:
                                    relationships[rel_key] = {'count': 0, 'funding': {}}
                                relationships[rel_key]['count'] += 1
                                
                                # Extract funding source for this document
                                if not funding_rows.empty:
                                    try:
                                        funding = funding_rows[doc_col].iloc[0]
                                        if funding and not pd.isna(funding):
                                            funding_sources[funding] = funding_sources.get(funding, 0) + 1
                                            
                                            # Add to relationship
                                            if funding not in relationships[rel_key]['funding']:
                                                relationships[rel_key]['funding'][funding] = 0
                                            relationships[rel_key]['funding'][funding] += 1
                                    except (IndexError, KeyError):
                                        pass
                        except (IndexError, KeyError):
                            pass
            except (IndexError, KeyError):
                pass
    
    # Get top 5 from each category
    top_pub_types = sorted(pub_types.items(), key=lambda x: x[1], reverse=True)[:5]
    top_study_designs = sorted(study_designs.items(), key=lambda x: x[1], reverse=True)[:5]
    top_funding_sources = sorted(funding_sources.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Create sets for quick lookup
    top_pub_type_names = {pt[0] for pt in top_pub_types}
    top_design_names = {d[0] for d in top_study_designs}
    top_funding_names = {f[0] for f in top_funding_sources}
    
    # Custom color scheme with Imperial Brands orange as the base
    colors = ['#FF7417', '#FF8C42', '#FFA15C', '#FFB676', '#FFCB91']
    
    # Define color mapping for publication types
    data_colors = {}
    for i, (pub_type, _) in enumerate(top_pub_types):
        data_colors[pub_type] = colors[i % len(colors)]
    
    # Build the hierarchical data structure
    data = []
    
    for i, (pub_type, pub_count) in enumerate(top_pub_types):
        pub_node = {
            "name": pub_type,
            "value": pub_count,  # Add count for single-level display
            "itemStyle": {
                "color": data_colors[pub_type]
            },
            "children": []
        }
        
        # Find study designs for this publication type (only top 5)
        for design_name, _ in top_study_designs:
            rel_key = f"{pub_type}|{design_name}"
            if rel_key in relationships:
                design_count = relationships[rel_key]['count']
                
                design_node = {
                    "name": design_name,
                    "value": design_count,  # Add count for level 2
                    "children": []
                }
                
                # Find funding sources for this combination (only top 5)
                funding_for_combo = relationships[rel_key]['funding']
                top_funding_for_combo = sorted(
                    [(k, v) for k, v in funding_for_combo.items() if k in top_funding_names],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # Limit to top 5 funding sources for this specific combination
                
                for funding_name, funding_count in top_funding_for_combo:
                    funding_node = {
                        "name": funding_name,
                        "value": funding_count
                    }
                    design_node["children"].append(funding_node)
                
                # Only add design node if it has funding children
                if design_node["children"]:
                    pub_node["children"].append(design_node)
        
        # Only add pub type node if it has study design children
        if pub_node["children"]:
            data.append(pub_node)
        else:
            # If no children but it's a top 5 pub type, add it anyway with empty children
            pub_node["children"] = []
            data.append(pub_node)
    
    return data

def create_pyecharts_sunburst_html(data):
    """
    Create a pyecharts sunburst chart and return HTML
    
    Parameters:
    data (list): Nested data structure for sunburst chart
    
    Returns:
    str: HTML content for the chart
    """
    # Custom background color (light orange tint)
    bg_color = '#ffebeb'
    
    # Create the Sunburst chart
    sunburst = (
        Sunburst(init_opts=opts.InitOpts(
            width="100%", 
            height="453px", 
            bg_color=bg_color,
            theme=ThemeType.LIGHT
        ))
        .add(
            series_name="Back",
            data_pair=data,
            highlight_policy="ancestor",
            radius=[0, "95%"],
            sort_="null",
            levels=[
                {},  # Level 0 - Center: "Research"
                {    # Level 1 - Publication Types (top 5)
                    "r0": "8%",
                    "r": "35%",
                    "label": {"rotate": "0", "fontSize": 10},
                    
                },
                {    # Level 2 - Study Designs (top 5)
                    "r0": "35%",
                    "r": "70%",
                    "label": {"rotate": "0", "fontSize": 10},
                },
                {    # Level 3 - Funding Sources (top 5)
                    "r0": "70%",
                    "r": "95%",
                    "label": {
                        "rotate": "0",
                        "fontSize": 9
                    }
                }
            ],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="E-Cigarette Research Overview",
                title_textstyle_opts=opts.TextStyleOpts(color="#333", font_size=16),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{b}: {c}"
            )
        )
    )
    
    # Create a temporary file to save the HTML
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmpfile:
        sunburst.render(tmpfile.name)
        html_path = tmpfile.name
    
    # Read the HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Clean up the temporary file
    os.unlink(html_path)
    
    return html_content

def display_pyecharts_sunburst(df, matching_docs):
    """
    Generate and display the pyecharts sunburst in Streamlit
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
    # Generate the data
    sunburst_data = generate_pyecharts_sunburst_data(df, matching_docs)
    
    if not sunburst_data:
        st.warning("Not enough data to generate the chart. Please adjust your filters.")
        return
    
    # Create the HTML
    html_content = create_pyecharts_sunburst_html(sunburst_data)
    
    # Display in Streamlit
    st.components.v1.html(html_content, height=470, scrolling=False)

def get_countries_by_study(df, matching_docs):
    """
    Extract countries mentioned in studies and count their occurrences.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the research data
    matching_docs (list): List of document column names that match current filters
    
    Returns:
    dict: Dictionary with countries as keys and their mention counts as values
    """
    country_data = {}
    
    # Find rows where Category is 'country_of_study'
    if 'Category' in df.columns and 'country_of_study' in df['Category'].values:
        country_rows = df[df['Category'] == 'country_of_study']
        
        for doc_col in matching_docs:
            country_value = country_rows[doc_col].iloc[0] if not country_rows.empty else None
            
            if country_value and not pd.isna(country_value):
                # Split by comma, semicolon, or 'and' to handle multiple countries in one cell
                split_countries = re.split(r',|\s+and\s+|;', str(country_value))
                
                for country in split_countries:
                    # Clean up country name
                    country = country.strip()
                    if country:
                        # Handle special cases for country names
                        if country.lower() in ['usa', 'us', 'u.s.', 'u.s.a.', 'united states']:
                            country = 'United States of America'
                        elif country.lower() in ['uk', 'u.k.', 'england', 'britain', 'great britain', 'united kingdon']:
                            country = 'United Kingdom'
                        
                        # Count occurrences
                        if country in country_data:
                            country_data[country] += 1
                        else:
                            country_data[country] = 1
    
    # Filter out 'Global' as it's not a country
    if 'Global' in country_data:
        del country_data['Global']
        
    return country_data

def create_country_choropleth(country_data):
    """
    Create a folium choropleth map based on country data.
    
    Parameters:
    country_data (dict): Dictionary with countries as keys and their mention counts as values
    
    Returns:
    folium.Map: A folium map with choropleth visualization
    """
    # Convert dictionary to DataFrame for easier handling
    df = pd.DataFrame(list(country_data.items()), columns=['Country', 'Count'])
    
    # Calculate percentiles for counts
    df['Percentile'] = df['Count'].rank(pct=True) * 100
    
    # Create a map centered on the world
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Add the GeoJSON with choropleth data
    choropleth = folium.Choropleth(
        geo_data="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
        name="Country Counts",
        data=df,
        columns=['Country', 'Percentile'],  # Use percentile instead of raw count
        key_on="feature.properties.name",  # Standard key for country name in GeoJSON
        fill_color="YlGn",  # Yellow to Green colormap
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Country Mentions (Percentile)",
    ).add_to(m)
    
    # Create a tooltip showing both raw count and percentile
    style_function = lambda x: {'fillColor': '#00000000', 'color': '#00000000'}
    highlight_function = lambda x: {'weight': 3, 'fillOpacity': 0.1}
    
    # Extract GeoJSON data for enhanced tooltips
    geojson = choropleth.geojson.data
    
    # Add custom tooltips showing both count and percentile
    folium.GeoJson(
        geojson,
        name='Count and Percentile',
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['name'],
            aliases=['Country:'],
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,
        ),
    ).add_to(m)
    
    # Add a custom legend showing percentile ranges
    legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    border-radius: 5px;">
            <p style="margin-bottom: 5px; font-weight: bold;">Percentile Ranges</p>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #f7fcb9; width: 20px; height: 20px; margin-right: 5px;"></div>
                <span>Low (0-33%)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #addd8e; width: 20px; height: 20px; margin-right: 5px;"></div>
                <span>Medium (34-66%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background-color: #31a354; width: 20px; height: 20px; margin-right: 5px;"></div>
                <span>High (67-100%)</span>
            </div>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def display_publication_type_chart(df, matching_docs, pub_df):
    """
    Create and display stacked chart for Publication Type by year
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    pub_df (pandas.DataFrame): DataFrame with publication data by year
    """
    # Create stacked chart for Publication Type by year
    pub_types_by_year = {}
    
    if 'publication_type' in df['Category'].values:
        year_rows = df[df['Category'] == 'publication_year']
        type_rows = df[df['Category'] == 'publication_type']
        
        for doc_col in matching_docs:
            year_value = year_rows[doc_col].iloc[0] if not year_rows.empty else None
            pub_type = type_rows[doc_col].iloc[0] if not type_rows.empty else None
            
            if year_value and pub_type and not pd.isna(year_value) and not pd.isna(pub_type):
                try:
                    year = int(float(year_value))
                    if year not in pub_types_by_year:
                        pub_types_by_year[year] = {}
                    
                    if pub_type in pub_types_by_year[year]:
                        pub_types_by_year[year][pub_type] += 1
                    else:
                        pub_types_by_year[year][pub_type] = 1
                        
                except (ValueError, TypeError):
                    continue
    
    if pub_types_by_year:
        # Get the top 5 publication types
        all_types = {}
        for year_data in pub_types_by_year.values():
            for pub_type, count in year_data.items():
                if pub_type in all_types:
                    all_types[pub_type] += count
                else:
                    all_types[pub_type] = count
        
        top_5_types = sorted(all_types.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_type_names = [t[0] for t in top_5_types]
        
        # Prepare data for plotting
        years = sorted(pub_types_by_year.keys())
        data_for_plot = []
        
        for year in years:
            year_total = sum(pub_types_by_year[year].values())
            row = {'Year': year, 'Total': year_total}
            
            # Add top 5 types
            for type_name in top_5_type_names:
                row[type_name] = pub_types_by_year[year].get(type_name, 0)
            
            # Add "Others" category
            others_count = 0
            for type_name, count in pub_types_by_year[year].items():
                if type_name not in top_5_type_names:
                    others_count += count
            
            row['Others'] = others_count
            data_for_plot.append(row)
        
        plot_df = pd.DataFrame(data_for_plot)
        
        # Create 100% stacked chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate percentages for each publication type
        for year_row in plot_df.to_dict('records'):
            year = year_row['Year']
            total = year_row['Total']
            running_total = 0
            
            # Add top 5 types
            for i, type_name in enumerate(top_5_type_names):
                value = year_row.get(type_name, 0)
                percentage = (value / total * 100) if total > 0 else 0
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[percentage],
                        name=type_name,
                        marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)],
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
                running_total += percentage
            
            # Add "Others" category
            others_pct = (year_row.get('Others', 0) / total * 100) if total > 0 else 0
            if others_pct > 0:
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[others_pct],
                        name="Others",
                        marker_color='lightgray',
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
        
        # Add total publications line chart (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=plot_df['Year'],
                y=plot_df['Total'],
                name="Total Publications",
                line=dict(color='red', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Publication Types by Year (Top 5)",
            barmode='stack',
            height=500,
            yaxis=dict(
                title="Percentage (%)",
                range=[0, 100]
            ),
            yaxis2=dict(
                title="Total Publications",
                titlefont=dict(color="red"),
                tickfont=dict(color="red")
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Publication type data not available for the filtered documents")

def display_funding_chart(df, matching_docs):
    """
    Create and display stacked chart for Funding Source by year
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
    # Create stacked chart for Funding Source by year
    funding_by_year = {}
    
    if 'type' in df['SubCategory'].values:
        year_rows = df[df['Category'] == 'publication_year']
        funding_rows = df[df['SubCategory'] == 'type']
        
        for doc_col in matching_docs:
            year_value = year_rows[doc_col].iloc[0] if not year_rows.empty else None
            funding = funding_rows[doc_col].iloc[0] if not funding_rows.empty else None
            
            if year_value and funding and not pd.isna(year_value) and not pd.isna(funding):
                try:
                    year = int(float(year_value))
                    if year not in funding_by_year:
                        funding_by_year[year] = {}
                    
                    if funding in funding_by_year[year]:
                        funding_by_year[year][funding] += 1
                    else:
                        funding_by_year[year][funding] = 1
                        
                except (ValueError, TypeError):
                    continue
    
    if funding_by_year:
        # Get the top 5 funding sources
        all_sources = {}
        for year_data in funding_by_year.values():
            for source, count in year_data.items():
                if source in all_sources:
                    all_sources[source] += count
                else:
                    all_sources[source] = count
        
        top_5_sources = sorted(all_sources.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_source_names = [s[0] for s in top_5_sources]
        
        # Prepare data for plotting
        years = sorted(funding_by_year.keys())
        data_for_plot = []
        
        for year in years:
            year_total = sum(funding_by_year[year].values())
            row = {'Year': year, 'Total': year_total}
            
            # Add top 5 sources
            for source_name in top_5_source_names:
                row[source_name] = funding_by_year[year].get(source_name, 0)
            
            # Add "Others" category
            others_count = 0
            for source_name, count in funding_by_year[year].items():
                if source_name not in top_5_source_names:
                    others_count += count
            
            row['Others'] = others_count
            data_for_plot.append(row)
        
        plot_df = pd.DataFrame(data_for_plot)
        
        # Create 100% stacked chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate percentages for each funding source
        for year_row in plot_df.to_dict('records'):
            year = year_row['Year']
            total = year_row['Total']
            
            # Add top 5 sources
            for i, source_name in enumerate(top_5_source_names):
                value = year_row.get(source_name, 0)
                percentage = (value / total * 100) if total > 0 else 0
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[percentage],
                        name=source_name,
                        marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)],
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
            
            # Add "Others" category
            others_pct = (year_row.get('Others', 0) / total * 100) if total > 0 else 0
            if others_pct > 0:
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[others_pct],
                        name="Others",
                        marker_color='lightgray',
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
        
        # Add total publications line chart (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=plot_df['Year'],
                y=plot_df['Total'],
                name="Total Publications",
                line=dict(color='red', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Funding Sources by Year (Top 5)",
            barmode='stack',
            height=500,
            yaxis=dict(
                title="Percentage (%)",
                range=[0, 100]
            ),
            yaxis2=dict(
                title="Total Publications",
                titlefont=dict(color="red"),
                tickfont=dict(color="red")
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Funding source data not available for the filtered documents")

def display_study_design_chart(df, matching_docs):
    """
    Create and display stacked chart for Study Design by year
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
    # Create stacked chart for Study Design by year
    design_by_year = {}
    
    if 'primary_type' in df['SubCategory'].values:
        year_rows = df[df['Category'] == 'publication_year']
        design_rows = df[df['SubCategory'] == 'primary_type']
        
        for doc_col in matching_docs:
            year_value = year_rows[doc_col].iloc[0] if not year_rows.empty else None
            design = design_rows[doc_col].iloc[0] if not design_rows.empty else None
            
            if year_value and design and not pd.isna(year_value) and not pd.isna(design):
                try:
                    year = int(float(year_value))
                    if year not in design_by_year:
                        design_by_year[year] = {}
                    
                    if design in design_by_year[year]:
                        design_by_year[year][design] += 1
                    else:
                        design_by_year[year][design] = 1
                        
                except (ValueError, TypeError):
                    continue
    
    if design_by_year:
        # Get the top 5 study designs
        all_designs = {}
        for year_data in design_by_year.values():
            for design, count in year_data.items():
                if design in all_designs:
                    all_designs[design] += count
                else:
                    all_designs[design] = count
        
        top_5_designs = sorted(all_designs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_design_names = [d[0] for d in top_5_designs]
        
        # Prepare data for plotting
        years = sorted(design_by_year.keys())
        data_for_plot = []
        
        for year in years:
            year_total = sum(design_by_year[year].values())
            row = {'Year': year, 'Total': year_total}
            
            # Add top 5 designs
            for design_name in top_5_design_names:
                row[design_name] = design_by_year[year].get(design_name, 0)
            
            # Add "Others" category
            others_count = 0
            for design_name, count in design_by_year[year].items():
                if design_name not in top_5_design_names:
                    others_count += count
            
            row['Others'] = others_count
            data_for_plot.append(row)
        
        plot_df = pd.DataFrame(data_for_plot)
        
        # Create 100% stacked chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate percentages for each study design
        for year_row in plot_df.to_dict('records'):
            year = year_row['Year']
            total = year_row['Total']
            
            # Add top 5 designs
            for i, design_name in enumerate(top_5_design_names):
                value = year_row.get(design_name, 0)
                percentage = (value / total * 100) if total > 0 else 0
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[percentage],
                        name=design_name,
                        marker_color=px.colors.qualitative.Dark2[i % len(px.colors.qualitative.Dark2)],
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
            
            # Add "Others" category
            others_pct = (year_row.get('Others', 0) / total * 100) if total > 0 else 0
            if others_pct > 0:
                fig.add_trace(
                    go.Bar(
                        x=[year],
                        y=[others_pct],
                        name="Others",
                        marker_color='lightgray',
                        showlegend=True if year == years[0] else False,
                        offsetgroup="A"
                    )
                )
        
        # Add total publications line chart (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=plot_df['Year'],
                y=plot_df['Total'],
                name="Total Publications",
                line=dict(color='red', width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Study Designs by Year (Top 5)",
            barmode='stack',
            height=500,
            yaxis=dict(
                title="Percentage (%)",
                range=[0, 100]
            ),
            yaxis2=dict(
                title="Total Publications",
                titlefont=dict(color="red"),
                tickfont=dict(color="red")
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Study design data not available for the filtered documents")

def display_country_map(df, matching_docs):
    """
    Create and display a choropleth map showing country data
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
    # Extract country data from matching documents
    country_data = get_countries_by_study(df, matching_docs)
    
    if country_data:
        # Create and display the map (full width)
        country_map = create_country_choropleth(country_data)
        st_folium(country_map, width=690, height=375)
        
        # Add collapsible section with top countries
        with st.expander("View Top Countries by Study Count", expanded=False):
            # Show a table of top countries with percentiles
            
            # Sort countries by count in descending order
            sorted_countries = sorted(country_data.items(), key=lambda x: x[1], reverse=True)
            
            # Create a formatted table
            table_data = []
            for i, (country, count) in enumerate(sorted_countries[:12], 1):
                # Calculate percentile rank
                table_data.append({
                    "Sr. No.": i,
                    "Country": country,
                    "Studies": count
                })
            
            # Display as a DataFrame
            table_df = pd.DataFrame(table_data)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            
            # Show total unique countries
            st.markdown(f"**Total unique countries in dataset**: {len(country_data)}")
    else:
        st.info("No country data available for the filtered documents.")

def display_yearly_chart(pub_df):
    """
    Display a simple bar chart of publications by year
    
    Parameters:
    pub_df (pandas.DataFrame): DataFrame with Year and Count columns
    """
    # Original yearly bar chart
    fig = px.bar(
        pub_df,
        x='Year',
        y='Count',
        title="Publications by Year",
        labels={'Count': 'Number of Publications', 'Year': 'Year'},
        color_discrete_sequence=['#f07300']
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def display_publication_distribution(df, matching_docs):
    """
    Main function to display the publication distribution visualizations
    based on the selected chart type.
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
    st.subheader("Publication Distribution")
    
    # Create radio buttons arranged horizontally for chart selection
    chart_type = st.radio(
        "Select Chart Type:",
        ["Overall", "Yearly", "Publication Type", "Funding Source", "Study Design"],
        horizontal=True
    )
    
    # Get publications by year data
    pub_df = get_publications_by_year(df, matching_docs)
    
    if not pub_df.empty:
        if chart_type == "Overall":
            display_pyecharts_sunburst(df, matching_docs)
            
        elif chart_type == "Yearly":
            display_yearly_chart(pub_df)
        
        elif chart_type == "Publication Type":
            display_publication_type_chart(df, matching_docs, pub_df)
        
        elif chart_type == "Funding Source":
            display_funding_chart(df, matching_docs)
        
        elif chart_type == "Study Design":
            display_study_design_chart(df, matching_docs)
        
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        
        
        
def render_harmful_ingredients_visualization(df, matching_docs):
    """
    Main function to display harmful ingredients visualization in Streamlit.
    
    Args:
        df: The main dataframe with all the data
        matching_docs: List of document columns that match the current filters
    """
    # Create a container for the visualization
    visualization_container = st.container()
    
    with visualization_container:
        # Main title
        st.subheader("Harmful Ingredients in E-Cigarettes by Evidence Strength")
        
        # Initialize session state variables if not exist
        if 'selected_ingredient' not in st.session_state:
            st.session_state.selected_ingredient = None
            
        # Extract ingredient data from the dataframe
        ingredients_data = extract_ingredients_data(df, matching_docs)
        
        if not ingredients_data:
            st.warning("No harmful ingredients data found in the selected documents.")
            return
            
        # Set default selected ingredient if none selected yet
        if st.session_state.selected_ingredient is None and ingredients_data:
            st.session_state.selected_ingredient = ingredients_data[0]['name']
        
        # Create the chart
        fig = create_ingredients_chart(ingredients_data)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        })
        
        # Add a selectbox below the chart for mobile or as an alternative to clicking
        all_ingredients = [item['name'] for item in ingredients_data]
        selected_index = all_ingredients.index(st.session_state.selected_ingredient) if st.session_state.selected_ingredient in all_ingredients else 0
        
        st.selectbox(
            "Select an ingredient to view health impacts:",
            all_ingredients,
            index=selected_index,
            key="ingredient_selector",
            on_change=update_selected_ingredient
        )
        
        # Display health impacts below the dropdown
        display_health_impacts(df, matching_docs, st.session_state.selected_ingredient)

def update_selected_ingredient():
    """Callback function for the selectbox to update the selected ingredient"""
    st.session_state.selected_ingredient = st.session_state.ingredient_selector

def extract_ingredients_data(df, matching_docs):
    """
    Extract harmful ingredients data from the dataframe with evidence strength breakdown.
    """
    # Find relevant rows
    ingredients_name_row = df[df['Category'] == 'harmful_ingredients'].loc[df['SubCategory'] == 'name']
    evidence_strength_row = df[df['Category'] == 'harmful_ingredients'].loc[df['SubCategory'] == 'evidence_strength']
    
    if ingredients_name_row.empty:
        return []
    
    # Function to parse ingredients list
    def parse_numbered_list(text):
        if not isinstance(text, str):
            return []
        
        # Split by numbered items pattern (e.g., "1) Item, 2) Item")
        parts = re.split(r'\d+\)\s+', text)
        
        # First part is usually empty because of the split pattern
        return [part.strip().rstrip(',') for part in parts[1:] if part.strip()]
    
    # Create a mapping of ingredients to their evidence strength by paper
    ingredient_data = {}
    
    # For each paper column that matches our filters
    for paper in matching_docs:
        if paper not in ingredients_name_row.columns:
            continue
            
        ingredients_text = ingredients_name_row[paper].iloc[0] if not ingredients_name_row.empty else None
        strengths_text = evidence_strength_row[paper].iloc[0] if not evidence_strength_row.empty else None
        
        if ingredients_text and pd.notna(ingredients_text):
            ingredients = parse_numbered_list(ingredients_text)
            strengths = parse_numbered_list(strengths_text) if strengths_text and pd.notna(strengths_text) else []
            
            # Match ingredients with their strengths (assuming they're in the same order)
            for i, ingredient in enumerate(ingredients):
                strength = strengths[i] if i < len(strengths) else 'Unknown'
                
                if ingredient not in ingredient_data:
                    ingredient_data[ingredient] = {
                        'name': ingredient,
                        'total': 0,
                        'Strong': 0,
                        'Moderate': 0,
                        'Weak': 0,
                        'Unknown': 0
                    }
                
                ingredient_data[ingredient]['total'] += 1
                ingredient_data[ingredient][strength] += 1
    
    # Convert to list and sort by total frequency
    sorted_data = sorted(ingredient_data.values(), key=lambda x: x['total'], reverse=True)
    
    return sorted_data

def create_ingredients_chart(ingredients_data):
    """
    Create a horizontal bar chart with stacked bars for evidence strength using pastel colors.
    """
    # Sort ingredients by total count
    ingredients = [item['name'] for item in ingredients_data]
    
    # Extract evidence strength values
    strong_values = [item['Strong'] for item in ingredients_data]
    moderate_values = [item['Moderate'] for item in ingredients_data]
    weak_values = [item['Weak'] for item in ingredients_data]
    unknown_values = [item['Unknown'] for item in ingredients_data]
    
    # Define pastel color palette
    pastel_colors = {
        'strong': '#FF9966  ',    # Orange
        'moderate': '#7ADCE0 ',  # Blue 
        'weak': '#F0E68C',      # Khaki 
        'unknown': '#A9A9A9'    # Dark gray 
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each evidence type with pastel colors
    fig.add_trace(go.Bar(
        y=ingredients,
        x=strong_values,
        name='Strong Evidence',
        orientation='h',
        marker=dict(color=pastel_colors['strong']),
        customdata=ingredients,
        hovertemplate='%{customdata}<br>Strong Evidence: %{x} papers<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=ingredients,
        x=moderate_values,
        name='Moderate Evidence',
        orientation='h',
        marker=dict(color=pastel_colors['moderate']),
        customdata=ingredients,
        hovertemplate='%{customdata}<br>Moderate Evidence: %{x} papers<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=ingredients,
        x=weak_values,
        name='Weak Evidence',
        orientation='h',
        marker=dict(color=pastel_colors['weak']),
        customdata=ingredients,
        hovertemplate='%{customdata}<br>Weak Evidence: %{x} papers<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=ingredients,
        x=unknown_values,
        name='Unknown Evidence',
        orientation='h',
        marker=dict(color=pastel_colors['unknown']),
        customdata=ingredients,
        hovertemplate='%{customdata}<br>Unknown Evidence: %{x} papers<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': '',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        barmode='stack',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            title='Number of Papers',
        ),
        yaxis=dict(
            title='',
            autorange='reversed',  # To match the original chart's order
            tickfont=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        height=500,
        hovermode='closest',
        plot_bgcolor='#ffebeb'  # White background for a cleaner look
    )
    
    return fig

def get_health_impacts(df, matching_docs, ingredient_name):
    """
    Get health impacts for a specific ingredient.
    """
    # Find health impact row
    health_impact_row = df[df['Category'] == 'harmful_ingredients'].loc[df['SubCategory'] == 'health_impact']
    ingredients_name_row = df[df['Category'] == 'harmful_ingredients'].loc[df['SubCategory'] == 'name']
    
    if health_impact_row.empty or ingredients_name_row.empty:
        return []
    
    # Function to parse ingredients and health impacts
    def parse_numbered_list(text):
        if not isinstance(text, str):
            return []
        
        parts = re.split(r'\d+\)\s+', text)
        return [part.strip().rstrip(',') for part in parts[1:] if part.strip()]
    
    # Get all health impacts for the specified ingredient
    impacts = []
    
    for paper in matching_docs:
        if paper not in ingredients_name_row.columns:
            continue
            
        ingredients_text = ingredients_name_row[paper].iloc[0] if not ingredients_name_row.empty else None
        impacts_text = health_impact_row[paper].iloc[0] if not health_impact_row.empty else None
        
        if ingredients_text and pd.notna(ingredients_text) and impacts_text and pd.notna(impacts_text):
            ingredients = parse_numbered_list(ingredients_text)
            health_impacts = parse_numbered_list(impacts_text)
            
            # Find the index of the ingredient
            for i, ingredient in enumerate(ingredients):
                if ingredient == ingredient_name and i < len(health_impacts):
                    impact = health_impacts[i]
                    if impact and impact not in impacts:
                        impacts.append(impact)
    
    return impacts

def display_health_impacts(df, matching_docs, ingredient_name):
    """
    Display health impacts for the selected ingredient using a minimal approach with bullet points.
    """
    # Get health impacts for the selected ingredient
    health_impacts = get_health_impacts(df, matching_docs, ingredient_name)
    
    # Display header
    st.subheader(f"Health Impacts: {ingredient_name}")
    
    # Display health impacts as bullet points
    if health_impacts:
        for impact in health_impacts:
            st.markdown(f"• {impact}")
    else:
        st.write("No specific health impact data available for this ingredient.")    
        
    
def render_perceived_benefits_visualization(df, matching_docs):
    st.subheader("Perceived Benefits Visualization")
    
    # Extract perceived health improvements data
    benefits_categories = [
        'sensory.smell', 'sensory.taste', 
        'physical.breathing', 'physical.physical_status', 'physical.stamina',
        'mental.mood', 'mental.sleep_quality'
    ]
    
    # Prepare data for visualization
    benefit_data = {}
    for benefit in benefits_categories:
        # Find rows with this benefit's overall percentage
        rows = df[(df['Category'] == 'perceived_health_improvements') & 
                  (df['SubCategory'] == f"{benefit}.overall_percentage")]
        
        values = []
        for doc_col in matching_docs:
            value = rows[doc_col].iloc[0] if not rows.empty else None
            if value and not pd.isna(value):
                try:
                    values.append(float(value))
                except:
                    continue
        
        if values:
            # Calculate average percentage reporting this benefit
            benefit_data[benefit.split('.')[-1]] = sum(values) / len(values)
    
    if benefit_data:
        # Create a radar/spider chart using Plotly
        categories = list(benefit_data.keys())
        values = list(benefit_data.values())
        
        # Create the radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Average Reported Benefits',
            line_color='#5aac90'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2 if values else 100]
                )
            ),
            title="Reported Health Improvement Areas (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display perceived benefits visualization")
    
    # Create a section for smoking cessation comparisons
    st.subheader("Smoking Cessation Success Rates")
    
    # Extract smoking cessation success rates
    cessation_rows = df[(df['Category'] == 'smoking_cessation') & 
                        (df['SubCategory'] == 'success_rates')]
    
    cessation_data = {}
    for doc_col in matching_docs:
        value = cessation_rows[doc_col].iloc[0] if not cessation_rows.empty else None
        if value and not pd.isna(value):
            cessation_data[doc_col] = value
    
    if cessation_data:
        # Create a bar chart for cessation success rates
        fig = px.bar(
            x=list(cessation_data.keys()),
            y=list(cessation_data.values()),
            labels={'x': 'Study', 'y': 'Success Rate (%)'},
            title="Smoking Cessation Success Rates Across Studies",
            color_discrete_sequence=['#5aac90']
        )
        
        fig.update_layout(
            xaxis_title="Study",
            yaxis_title="Success Rate (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display smoking cessation visualization")
        
        

def render_research_trends_visualization(df, matching_docs):
    # Extract study design types over time
    st.subheader("Evolution of Research Methodologies")
    
    # Get primary study types
    study_type_rows = df[df['SubCategory'] == 'primary_type']
    
    # Get publication years for matching documents
    year_rows = df[df['Category'] == 'publication_year']
    
    # Create a dictionary to store study types by year
    study_types_by_year = {}
    
    for doc_col in matching_docs:
        # Get year for this document
        year_value = year_rows[doc_col].iloc[0] if not year_rows.empty else None
        study_type = study_type_rows[doc_col].iloc[0] if not study_type_rows.empty else None
        
        if year_value and study_type and not pd.isna(year_value) and not pd.isna(study_type):
            try:
                year = int(float(year_value))
                
                if year not in study_types_by_year:
                    study_types_by_year[year] = {}
                
                if study_type not in study_types_by_year[year]:
                    study_types_by_year[year][study_type] = 0
                    
                study_types_by_year[year][study_type] += 1
            except:
                continue
    
    if study_types_by_year:
        # Prepare data for stacked bar chart
        years = sorted(study_types_by_year.keys())
        study_types = set()
        for year_data in study_types_by_year.values():
            for study_type in year_data.keys():
                study_types.add(study_type)
        
        # Create data for each study type
        data = []
        for study_type in study_types:
            study_type_data = []
            for year in years:
                if year in study_types_by_year and study_type in study_types_by_year[year]:
                    study_type_data.append(study_types_by_year[year][study_type])
                else:
                    study_type_data.append(0)
            
            data.append(go.Bar(
                name=study_type,
                x=years,
                y=study_type_data
            ))
        
        # Create stacked bar chart
        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title="Study Design Distribution by Year",
            xaxis_title="Year",
            yaxis_title="Number of Studies",
            height=550,
            legend_title="Study Design"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display study design evolution visualization")
        
        
def render_contradictions_visualization(df, matching_docs):
    st.subheader("Contradiction Analysis")
    
    # Extract data about contradictions
    contradictions_rows = df[(df['Category'] == 'contradictions') & 
                            (df['SubCategory'] == 'conflicts_with_literature')]
    
    # Count documents with contradictions
    contradictions_count = 0
    no_contradictions_count = 0
    
    for doc_col in matching_docs:
        contradiction_value = contradictions_rows[doc_col].iloc[0] if not contradictions_rows.empty else None
        
        if contradiction_value and not pd.isna(contradiction_value):
            # Check if there's any indication of contradictions
            if any(keyword in str(contradiction_value).lower() for keyword in 
                  ['conflict', 'contradict', 'inconsistent', 'differs', 'contrary', 'opposed']):
                contradictions_count += 1
            else:
                no_contradictions_count += 1
    
    # Create a pie chart showing proportion of studies with contradictions
    if contradictions_count + no_contradictions_count > 0:
        fig = go.Figure(data=[go.Pie(
            labels=['Studies with contradictions', 'Studies without contradictions'],
            values=[contradictions_count, no_contradictions_count],
            hole=.4,
            marker_colors=['#FF7417', '#5aac90']
        )])
        
        fig.update_layout(
            title="Proportion of Studies with Literature Contradictions",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to analyze contradictions")
    
    # Extract evidence strength for key claims
    st.subheader("Evidence Strength Analysis")
    
    # Categories with evidence strength data
    evidence_categories = [
        ('chemicals_implicated', 'evidence_strength'),
        ('biological_pathways', 'evidence_strength'),
        ('comparative_benefits.vs_traditional_cigarettes', 'evidence_strength')
    ]
    
    evidence_data = {}
    
    for category, subcategory in evidence_categories:
        # Find rows with evidence strength
        if category.startswith('comparative_benefits'):
            category_main, category_sub = category.split('.')
            rows = df[(df['Category'] == category_main) & 
                      (df['SubCategory'] == f"{category_sub}.{subcategory}")]
        else:
            rows = df[(df['Category'] == category) & 
                      (df['SubCategory'] == subcategory)]
        
        values = []
        for doc_col in matching_docs:
            value = rows[doc_col].iloc[0] if not rows.empty else None
            if value and not pd.isna(value):
                values.append(str(value))
        
        if values:
            # Categorize evidence levels
            evidence_levels = {
                'strong': 0,
                'moderate': 0,
                'weak': 0,
                'inconclusive': 0
            }
            
            for value in values:
                value_lower = value.lower()
                if 'strong' in value_lower:
                    evidence_levels['strong'] += 1
                elif 'moderate' in value_lower:
                    evidence_levels['moderate'] += 1
                elif 'weak' in value_lower:
                    evidence_levels['weak'] += 1
                else:
                    evidence_levels['inconclusive'] += 1
            
            # Format category name for display
            display_name = category.replace('_', ' ').replace('.', ' - ')
            evidence_data[display_name] = evidence_levels
    
    if evidence_data:
        # Create a grouped bar chart for evidence strength
        categories = list(evidence_data.keys())
        strong_values = [evidence_data[cat]['strong'] for cat in categories]
        moderate_values = [evidence_data[cat]['moderate'] for cat in categories]
        weak_values = [evidence_data[cat]['weak'] for cat in categories]
        inconclusive_values = [evidence_data[cat]['inconclusive'] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(name='Strong', x=categories, y=strong_values, marker_color='#5aac90'),
            go.Bar(name='Moderate', x=categories, y=moderate_values, marker_color='#88c9b3'),
            go.Bar(name='Weak', x=categories, y=weak_values, marker_color='#ffbc79'),
            go.Bar(name='Inconclusive', x=categories, y=inconclusive_values, marker_color='#FF7417')
        ])
        
        fig.update_layout(
            barmode='group',
            title="Evidence Strength Across Research Areas",
            xaxis_title="Research Area",
            yaxis_title="Number of Studies",
            height=500,
            legend_title="Evidence Level"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display evidence strength visualization")
        
        
        
def render_bias_visualization(df, matching_docs):
    st.subheader("Research Methodology Assessment")
    
    # Create radio buttons for switching between visualizations without extra space
    viz_option = st.radio(
        "Select Visualization",
        ["Funding Sources", "Bias Assessment", "Funding and Conclusions"],
        horizontal=True,
        label_visibility="visible"
    )
    
    # Extract funding source information
    funding_rows = df[df['SubCategory'] == 'type']
    
    funding_types = {}
    for doc_col in matching_docs:
        funding_type = funding_rows[doc_col].iloc[0] if not funding_rows.empty else None
        if funding_type and not pd.isna(funding_type):
            if funding_type not in funding_types:
                funding_types[funding_type] = 0
            funding_types[funding_type] += 1
    
    # Extract bias assessment data
    bias_categories = [
        'selection_bias', 'measurement_bias', 'confounding_factors', 
        'attrition_bias', 'reporting_bias'
    ]
    
    bias_data = {}
    
    for bias_type in bias_categories:
        # Find rows with this bias type
        rows = df[df['Category'] == bias_type]
        
        values = []
        for doc_col in matching_docs:
            value = rows[doc_col].iloc[0] if not rows.empty else None
            if value and not pd.isna(value):
                values.append(str(value))
        
        if values:
            # Categorize bias levels
            bias_levels = {
                'high': 0,
                'moderate': 0,
                'low': 0,
                'unclear': 0
            }
            
            for value in values:
                value_lower = value.lower()
                if 'high' in value_lower or 'significant' in value_lower:
                    bias_levels['high'] += 1
                elif 'moderate' in value_lower:
                    bias_levels['moderate'] += 1
                elif 'low' in value_lower or 'minimal' in value_lower:
                    bias_levels['low'] += 1
                else:
                    bias_levels['unclear'] += 1
            
            bias_data[bias_type.replace('_', ' ')] = bias_levels
    
    # Extract main conclusions for sentiment analysis
    conclusions_rows = df[df['Category'] == 'main_conclusions']
    
    # Analyze sentiment of conclusions by funding source
    conclusion_sentiment = {}
    
    for doc_col in matching_docs:
        funding_type = funding_rows[doc_col].iloc[0] if not funding_rows.empty else None
        conclusion = conclusions_rows[doc_col].iloc[0] if not conclusions_rows.empty else None
        
        if funding_type and conclusion and not pd.isna(funding_type) and not pd.isna(conclusion):
            # Simple sentiment analysis
            conclusion_lower = str(conclusion).lower()
            sentiment = 'neutral'
            
            positive_terms = ['beneficial', 'positive', 'improvement', 'effective', 'better', 'safe']
            negative_terms = ['harmful', 'negative', 'risk', 'adverse', 'danger', 'concern']
            
            if any(term in conclusion_lower for term in positive_terms):
                sentiment = 'positive'
            elif any(term in conclusion_lower for term in negative_terms):
                sentiment = 'negative'
            
            if funding_type not in conclusion_sentiment:
                conclusion_sentiment[funding_type] = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            conclusion_sentiment[funding_type][sentiment] += 1
    
    # Now show the selected visualization based on radio button choice
    if viz_option == "Funding Sources":
        if funding_types:
            # Create pie chart for funding sources with pastel colors
            fig = px.pie(
                values=list(funding_types.values()),
                names=list(funding_types.keys()),
                title="Funding Sources Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel1  # Changed to pastel colors
            )
            
            # Apply margin settings in update_layout instead of directly in px.pie
            fig.update_layout(
                height=430,
                margin=dict(t=40, b=0, l=0, r=0)  # Reduce top margin to remove space
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display funding source visualization")
            
    elif viz_option == "Bias Assessment":
        if bias_data:
            # Create a heatmap for bias assessment
            bias_types = list(bias_data.keys())
            bias_levels = ['low', 'moderate', 'high', 'unclear']
            
            z_data = []
            for level in bias_levels:
                z_data.append([bias_data[bias_type][level] for bias_type in bias_types])
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=bias_types,
                y=bias_levels,
                colorscale=[
                    [0, '#5aac90'],  # low values (good)
                    [0.33, '#88c9b3'],
                    [0.66, '#ffbc79'],
                    [1, '#FFA589']   # high values (concerning) - changed to lighter orange
                ],
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Bias Assessment Across Studies",
                xaxis_title="Bias Type",
                yaxis_title="Bias Level",
                height=430,
                margin=dict(t=40, b=0, l=0, r=0)  # Reduce top margin to remove space
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display bias assessment visualization")
            
    elif viz_option == "Funding and Conclusions":
        if conclusion_sentiment:
            # Prepare data for grouped bar chart
            funding_sources = list(conclusion_sentiment.keys())
            positive_values = [conclusion_sentiment[source]['positive'] for source in funding_sources]
            neutral_values = [conclusion_sentiment[source]['neutral'] for source in funding_sources]
            negative_values = [conclusion_sentiment[source]['negative'] for source in funding_sources]
            
            fig = go.Figure(data=[
                go.Bar(name='Positive', x=funding_sources, y=positive_values, marker_color='#5aac90'),
                go.Bar(name='Neutral', x=funding_sources, y=neutral_values, marker_color='#88c9b3'),
                go.Bar(name='Negative', x=funding_sources, y=negative_values, marker_color='#FFA589')
            ])
            
            fig.update_layout(
                barmode='group',
                title="Conclusion Sentiment by Funding Source",
                xaxis_title="Funding Source",
                yaxis_title="Number of Studies",
                height=430,
                legend_title="Sentiment",
                margin=dict(t=40, b=0, l=0, r=0)  # Reduce top margin to remove space
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to analyze conclusion sentiment by funding source")
            
        


def render_publication_level_visualization(df, matching_docs):
    st.subheader("Research Distribution by Geography and Type")
    
    # Create radio buttons to toggle between visualizations without extra space
    visualization_type = st.radio(
        "Select Visualization",
        ["Geographic Distribution", "Publication Types"],
        horizontal=True,
        label_visibility="visible"
    )
    
    if visualization_type == "Geographic Distribution":
        # Use the provided function
        display_country_map(df, matching_docs)
    
    elif visualization_type == "Publication Types":
        # Extract publication types
        publication_type_rows = df[df['Category'] == 'publication_type']
        
        publication_types = {}
        for doc_col in matching_docs:
            pub_type = publication_type_rows[doc_col].iloc[0] if not publication_type_rows.empty else None
            if pub_type and not pd.isna(pub_type):
                if pub_type not in publication_types:
                    publication_types[pub_type] = 0
                publication_types[pub_type] += 1
        
        if publication_types:
            # Create a pie chart for publication types with pastel colors
            fig = px.pie(
                names=list(publication_types.keys()),
                values=list(publication_types.values()),
                title="Distribution of Publication Types",
                color_discrete_sequence=px.colors.qualitative.Pastel1,
            )
            
            fig.update_layout(
                height=380,
                margin=dict(t=40, b=0, l=0, r=0)  # Reduce top margin to remove space
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display publication type visualization")

        

def display_sankey_dropdown(categories_dict, tab_title, height=500, bg_color="#fff2f2", right="15%"):
    """
    Display a collapsible dropdown with a Sankey chart visualization
    with improved handling of duplicate node names
    
    Args:
        categories_dict: Dictionary of categories and subcategories
        tab_title: Title of the current tab
        height: Height of the chart in pixels
        bg_color: Background color for the expander (default: light gray)
        right: CSS position for the right margin
    """

    
    # Add custom CSS for the expander background
    custom_css = f"""
    <style>
        /* Target all expander elements */
        div[data-testid="stExpander"] {{
            background-color: {bg_color} !important;
        }}
        /* Target the header of the expander */
        div[data-testid="stExpander"] > div:first-child {{
            background-color: {bg_color} !important;
        }}
        /* Target the content area of the expander */
        div[data-testid="stExpander"] > div:nth-child(2) {{
            background-color: {bg_color} !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    with st.expander(f"View Source Data for Generated {tab_title} Insights", expanded=False):
        # Get the category count and total field count for context
        category_count = len(categories_dict)
        field_count = sum(len(subcategories) for subcategories in categories_dict.values())
        
        # Display context information
        st.caption(f"Analysis based on {category_count} main categories and total {field_count} attributes")
        
        # Function to format display names
        def format_display_name(name):
            """Format a name for display by removing underscores and capitalizing"""
            # Handle general case
            if name.endswith('_general'):
                base_name = name[:-8]  # Remove '_general'
                return f"{base_name.replace('_', ' ').title()}"
            return name.replace('_', ' ').title()
        
        # Generate pastel colors for each category
        def generate_pastel_colors(n):
            """
            Generate n visually distinct pastel colors
            
            Args:
                n: Number of colors to generate
                
            Returns:
                List of pastel color hex codes
            """
            # Predefined set of pastel colors
            pastel_colors = [
                '#8de9c8',  # mint
                '#a2b0e6',  # periwinkle
                '#ffc298',  # peach
                '#d2eda5',  # light green
                '#b0ebeb',  # pale cyan
                '#fdb7bf',  # baby pink
                '#bde4dd',  # pale teal
                '#b4d8f3',  # pale blue
                '#fbfba0',  # pale yellow
                '#d1bbf0',  # pale purple
                '#a4cbd0',  # slate blue
                '#ffb9b9',  # pale red
                '#a4fcb1',  # pale lime
            ]
            
            # If we need more colors than in our predefined list, generate more
            if n > len(pastel_colors):
                for i in range(n - len(pastel_colors)):
                    # Generate a random pastel color
                    r = random.randint(180, 240)
                    g = random.randint(180, 240)
                    b = random.randint(180, 240)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    pastel_colors.append(color)
            
            return pastel_colors[:n]
        
        # Create nodes and links for Sankey chart
        nodes = []
        links = []
        
        # Use a mapping structure to track nodes and their display names
        # Format: {"internal_id": {"index": node_index, "display": display_name}}
        node_map = {}
        
        # Track hierarchy paths for proper node connection
        hierarchy_paths = {}
        
        category_colors = generate_pastel_colors(len(categories_dict))
        
        # Add "Research Data" as root node
        root_color = '#FFB7B2'  # Soft pink for root
        root_id = "root"
        nodes.append({"name": root_id, "value": "Research Data", "itemStyle": {"color": root_color}})
        node_map[root_id] = {"index": 0, "display": "Research Data"}
        current_index = 1
        
        # First pass: Create category nodes
        for idx, (category, subcategories) in enumerate(categories_dict.items()):
            category_color = category_colors[idx]
            category_display_name = format_display_name(category)
            category_id = f"cat_{idx}"
            
            # Add category node
            nodes.append({
                "name": category_id,
                "value": category_display_name,
                "itemStyle": {"color": category_color}
            })
            
            # Store node info
            node_map[category_id] = {"index": current_index, "display": category_display_name}
            hierarchy_paths[category] = category_id
            
            # Add link from root to category
            links.append({
                "source": node_map[root_id]["index"],
                "target": current_index,
                "value": len(subcategories),
                "lineStyle": {"color": "#f8d6d5"}
            })
            
            current_index += 1
        
        # Second pass: Process subcategories
        for category_idx, (category, subcategories) in enumerate(categories_dict.items()):
            category_color = category_colors[category_idx]
            category_id = hierarchy_paths[category]
            
            for subcategory_idx, subcategory in enumerate(subcategories):
                if '.' in subcategory:
                    # Handle hierarchical subcategory (contains dots)
                    parts = subcategory.split('.')
                    current_category = category
                    parent_id = category_id
                    
                    # Process each part of the path
                    for part_idx, part in enumerate(parts):
                        # Create a unique identifier for this node in this specific path
                        path_so_far = f"{current_category}.{'.'.join(parts[:part_idx+1])}"
                        internal_id = f"node_{category_idx}_{subcategory_idx}_{part_idx}"
                        display_name = format_display_name(part)
                        
                        # Check if this exact path has already been created
                        if path_so_far not in hierarchy_paths:
                            # Create new node
                            nodes.append({
                                "name": internal_id,
                                "value": display_name,
                                "itemStyle": {"color": category_color}
                            })
                            
                            # Add link from parent
                            links.append({
                                "source": node_map[parent_id]["index"],
                                "target": current_index,
                                "value": 1,
                                "lineStyle": {"color": category_color}
                            })
                            
                            # Update mappings
                            node_map[internal_id] = {"index": current_index, "display": display_name}
                            hierarchy_paths[path_so_far] = internal_id
                            
                            current_index += 1
                        
                        # Set parent for next iteration
                        parent_id = hierarchy_paths[path_so_far]
                        current_category = path_so_far
                else:
                    # Handle simple subcategory
                    display_name = format_display_name(subcategory)
                    path_id = f"{category}.{subcategory}"
                    internal_id = f"node_{category_idx}_{subcategory_idx}"
                    
                    # Check if this exact path already exists
                    if path_id not in hierarchy_paths:
                        # Add node
                        nodes.append({
                            "name": internal_id,
                            "value": display_name,
                            "itemStyle": {"color": category_color}
                        })
                        
                        # Add link from category to subcategory
                        links.append({
                            "source": node_map[category_id]["index"],
                            "target": current_index,
                            "value": 1,
                            "lineStyle": {"color": category_color}
                        })
                        
                        # Update mappings
                        node_map[internal_id] = {"index": current_index, "display": display_name}
                        hierarchy_paths[path_id] = internal_id
                        
                        current_index += 1
        
        # Prepare final data for the chart
        sankey_data = {"nodes": nodes, "links": links}
        
        # Create a unique ID for the chart
        chart_id = f"sankey-chart-{tab_title.replace(' ', '-').lower()}"
        
        # HTML and JavaScript for the chart
        html_content = f"""
        <div id="{chart_id}" style="width:100%; height:{height}px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <script>
            // Function to initialize chart
            function initChart() {{
                var chartDom = document.getElementById('{chart_id}');
                if (!chartDom) {{
                    // Chart container not found, retry after a delay
                    setTimeout(initChart, 300);
                    return;
                }}
                var myChart = echarts.init(chartDom);
                var option = {{
                    title: {{
                        text: '',
                        left: 'center'
                    }},
                    tooltip: {{
                        show: false
                    }},
                    series: [
                        {{
                            type: 'sankey',
                            right: '{right}',
                            data: {json.dumps(sankey_data["nodes"])},
                            links: {json.dumps(sankey_data["links"])},
                            emphasis: {{
                                focus: 'adjacency'
                            }},
                            nodeAlign: 'justify',
                            label: {{
                                color: 'black',
                                fontWeight: 400,
                                opacity: 0.7,
                                formatter: function(params) {{
                                    // Use the 'value' property as the display name
                                    return params.data.value || params.name;
                                }}
                            }},
                            lineStyle: {{
                                curveness: 0.5,
                                opacity: 0.4
                            }}
                        }}
                    ]
                }};
                // Apply options
                myChart.setOption(option);
                // Handle resize
                window.addEventListener('resize', function() {{
                    if (myChart) {{
                        myChart.resize();
                    }}
                }});
            }}
            // Start initialization when document is ready
            document.addEventListener('DOMContentLoaded', initChart);
            // Also try to init immediately in case DOM is already loaded
            initChart();
        </script>
        """
        # Display the chart using components.html
        st.components.v1.html(html_content, height=height+20)
        
        

def display_main_category_sankey(categories_df, selected_main_category, height=500, bg_color="#fff2f2", right="15%"):
    """
    Display a Sankey chart for a selected main category without an expander
    
    Args:
        categories_df: DataFrame containing the category structure data
        selected_main_category: The main category selected by the user
        height: Height of the chart in pixels
        bg_color: Background color for the chart area
        right: CSS position for the right margin
    """
    import json
    import random
    
    # Filter the dataframe for the selected main category
    filtered_df = categories_df[categories_df["Main Category"] == selected_main_category]
    
    # Collect all subcategories
    categories_dict = {}
    for _, row in filtered_df.iterrows():
        category = row["Category"]
        subcategory = row["SubCategory"]
        
        if pd.notna(category):
            if category not in categories_dict:
                categories_dict[category] = []
            
            # Handle cases where subcategory is '-' or NaN
            if pd.notna(subcategory) and subcategory != '-':
                categories_dict[category].append(subcategory)
            else:
                # For '-' or NaN, create a generic subcategory
                generic_name = f"{category}_general"
                categories_dict[category].append(generic_name)
    
    # Function to format display names
    def format_display_name(name):
        """Format a name for display by removing underscores and capitalizing"""
        # Handle general case
        if name.endswith('_general'):
            base_name = name[:-8]  # Remove '_general'
            return f"{base_name.replace('_', ' ').title()}"
        return name.replace('_', ' ').title()
    
    # Generate pastel colors for each category
    def generate_pastel_colors(n):
        pastel_colors = [
            '#8de9c8',  # mint
            '#a2b0e6',  # periwinkle
            '#ffc298',  # peach
            '#d2eda5',  # light green
            '#b0ebeb',  # pale cyan
            '#fdb7bf',  # baby pink
            '#bde4dd',  # pale teal
            '#b4d8f3',  # pale blue
            '#fbfba0',  # pale yellow
            '#d1bbf0',  # pale purple
            '#a4cbd0',  # slate blue
            '#ffb9b9',  # pale red
            '#a4fcb1',  # pale lime
        ]
        
        if n > len(pastel_colors):
            for i in range(n - len(pastel_colors)):
                r = random.randint(180, 240)
                g = random.randint(180, 240)
                b = random.randint(180, 240)
                color = f'#{r:02x}{g:02x}{b:02x}'
                pastel_colors.append(color)
        
        return pastel_colors[:n]
    
    # Create nodes and links for Sankey chart
    nodes = []
    links = []
    
    # Use a mapping structure to track nodes and their display names
    # Format: {"internal_id": {"index": node_index, "display": display_name}}
    node_map = {}
    
    # Track hierarchy paths for proper node connection
    hierarchy_paths = {}
    
    category_colors = generate_pastel_colors(len(categories_dict))
    
    # Add main category as root node
    root_color = '#FFB7B2'  # Soft pink for root
    root_id = "root"
    nodes.append({"name": root_id, "value": selected_main_category, "itemStyle": {"color": root_color}})
    node_map[root_id] = {"index": 0, "display": selected_main_category}
    current_index = 1
    
    # First pass: Create category nodes
    for idx, (category, subcategories) in enumerate(categories_dict.items()):
        category_color = category_colors[idx]
        category_display_name = format_display_name(category)
        category_id = f"cat_{idx}"
        
        # Add category node
        nodes.append({
            "name": category_id,
            "value": category_display_name,
            "itemStyle": {"color": category_color}
        })
        
        # Store node info
        node_map[category_id] = {"index": current_index, "display": category_display_name}
        hierarchy_paths[category] = category_id
        
        # Add link from root to category
        links.append({
            "source": node_map[root_id]["index"],
            "target": current_index,
            "value": len(subcategories),
            "lineStyle": {"color": "#f8d6d5"}
        })
        
        current_index += 1
    
    # Second pass: Process subcategories
    for category_idx, (category, subcategories) in enumerate(categories_dict.items()):
        category_color = category_colors[category_idx]
        category_id = hierarchy_paths[category]
        
        for subcategory_idx, subcategory in enumerate(subcategories):
            if '.' in subcategory:
                # Handle hierarchical subcategory (contains dots)
                parts = subcategory.split('.')
                current_category = category
                parent_id = category_id
                
                # Process each part of the path
                for part_idx, part in enumerate(parts):
                    # Create a unique identifier for this node in this specific path
                    path_so_far = f"{current_category}.{'.'.join(parts[:part_idx+1])}"
                    internal_id = f"node_{category_idx}_{subcategory_idx}_{part_idx}"
                    display_name = format_display_name(part)
                    
                    # Check if this exact path has already been created
                    if path_so_far not in hierarchy_paths:
                        # Create new node
                        nodes.append({
                            "name": internal_id,
                            "value": display_name,
                            "itemStyle": {"color": category_color}
                        })
                        
                        # Add link from parent
                        links.append({
                            "source": node_map[parent_id]["index"],
                            "target": current_index,
                            "value": 1,
                            "lineStyle": {"color": category_color}
                        })
                        
                        # Update mappings
                        node_map[internal_id] = {"index": current_index, "display": display_name}
                        hierarchy_paths[path_so_far] = internal_id
                        
                        current_index += 1
                    
                    # Set parent for next iteration
                    parent_id = hierarchy_paths[path_so_far]
                    current_category = path_so_far
            else:
                # Handle simple subcategory
                display_name = format_display_name(subcategory)
                path_id = f"{category}.{subcategory}"
                internal_id = f"node_{category_idx}_{subcategory_idx}"
                
                # Check if this exact path already exists
                if path_id not in hierarchy_paths:
                    # Add node
                    nodes.append({
                        "name": internal_id,
                        "value": display_name,
                        "itemStyle": {"color": category_color}
                    })
                    
                    # Add link from category to subcategory
                    links.append({
                        "source": node_map[category_id]["index"],
                        "target": current_index,
                        "value": 1,
                        "lineStyle": {"color": category_color}
                    })
                    
                    # Update mappings
                    node_map[internal_id] = {"index": current_index, "display": display_name}
                    hierarchy_paths[path_id] = internal_id
                    
                    current_index += 1
    
    # Prepare final data for the chart
    sankey_data = {"nodes": nodes, "links": links}
    
    # Get counts for context
    category_count = len(categories_dict)
    field_count = sum(len(subcategories) for subcategories in categories_dict.values())
    
    # Display context information
    st.caption(f"<span style='margin-left: 10px;'>Analysis based on {category_count} categories and total {field_count} attributes</span>", unsafe_allow_html=True)    
    
    # Create a unique ID for the chart
    chart_id = f"sankey-chart-{selected_main_category.replace(' ', '-').lower()}"
    
    # HTML and JavaScript for the chart
    html_content = f"""
    <div id="{chart_id}" style="width:100%; height:{height}px; background-color:{bg_color};"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script>
        // Function to initialize chart
        function initChart() {{
            var chartDom = document.getElementById('{chart_id}');
            if (!chartDom) {{
                // Chart container not found, retry after a delay
                setTimeout(initChart, 300);
                return;
            }}
            var myChart = echarts.init(chartDom);
            var option = {{
                title: {{
                    text: '',
                    left: 'center'
                }},
                tooltip: {{
                    show: false
                }},
                series: [
                    {{
                        type: 'sankey',
                        right: '{right}',
                        data: {json.dumps(sankey_data["nodes"])},
                        links: {json.dumps(sankey_data["links"])},
                        emphasis: {{
                            focus: 'adjacency'
                        }},
                        nodeAlign: 'justify',
                        label: {{
                            color: 'black',
                            fontWeight: 400,
                            opacity: 0.7,
                            formatter: function(params) {{
                                // Use the 'value' property as the display name
                                return params.data.value || params.name;
                            }}
                        }},
                        lineStyle: {{
                            curveness: 0.5,
                            opacity: 0.4
                        }}
                    }}
                ]
            }};
            // Apply options
            myChart.setOption(option);
            // Handle resize
            window.addEventListener('resize', function() {{
                if (myChart) {{
                    myChart.resize();
                }}
            }});
        }}
        // Start initialization when document is ready
        document.addEventListener('DOMContentLoaded', initChart);
        // Also try to init immediately in case DOM is already loaded
        initChart();
    </script>
    """
    # Display the chart using components.html
    st.components.v1.html(html_content, height=height+20)