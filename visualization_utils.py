import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import re
import tempfile
import os

from pyecharts import options as opts
from pyecharts.charts import Sunburst
from pyecharts.globals import ThemeType

import folium
from streamlit_folium import st_folium

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
            
            # Calculate percentiles for display
            country_counts = [count for _, count in sorted_countries]
            total_countries = len(country_counts)
            
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
        ["Overall", "Yearly", "Pub. Type", "Funding", "Study Design", "Country"],
        horizontal=True
    )
    
    # Get publications by year data
    pub_df = get_publications_by_year(df, matching_docs)
    
    if not pub_df.empty:
        if chart_type == "Overall":
            display_pyecharts_sunburst(df, matching_docs)
            
        elif chart_type == "Yearly":
            display_yearly_chart(pub_df)
        
        elif chart_type == "Pub. Type":
            display_publication_type_chart(df, matching_docs, pub_df)
        
        elif chart_type == "Funding":
            display_funding_chart(df, matching_docs)
        
        elif chart_type == "Study Design":
            display_study_design_chart(df, matching_docs)
        
        elif chart_type == "Country":
            display_country_map(df, matching_docs)
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")