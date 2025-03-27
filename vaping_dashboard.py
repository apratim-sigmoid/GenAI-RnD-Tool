import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from openai import OpenAI

from pyecharts import options as opts
from pyecharts.charts import Sunburst
from pyecharts.globals import ThemeType
import tempfile
import os

# Page config
st.set_page_config(
    page_title="IB GenAI R&D Tool",
    page_icon="🔬",
    layout="wide"
)



def extract_research_insights_from_docs(df, matching_docs):
    """
    Extract comprehensive research insights from matching documents including key findings,
    causal mechanisms, R&D insights, study characteristics, and health outcomes.
    """
    insights = {}
    
    # Categories to extract
    categories_to_extract = {
        "Key Findings": [
            "main_conclusions", "primary_outcomes", "secondary_outcomes", 
            "novel_findings", "limitations", "generalizability", 
            "future_research_suggestions", "contradictions"
        ],
        "Causal Mechanisms": [
            "chemicals_implicated", "biological_pathways", 
            "device_factors", "usage_pattern_factors"
        ],
        "R&D Insights": [
            "harmful_ingredients", "device_design_implications", 
            "comparative_benefits", "potential_innovation_areas",
            "operating_parameters"
        ],
        "Study Characteristics": [
            "primary_type", "secondary_features", "time_periods",
            "total_size", "user_groups", "e_cigarette_specifications",
            "data_collection_method"
        ],
        "Health Outcomes": [
            "respiratory_effects", "cardiovascular_effects", "oral_health",
            "neurological_effects", "psychiatric_effects", "cancer_risk",
            "developmental_effects", "other_health_outcomes"
        ]
    }
    
    # For each matching document, extract the insights
    for doc_col in matching_docs:
        doc_insights = {}
        
        # Get title if available
        title = None
        title_row = df[(df['Main Category'] == 'meta_data') & (df['Category'] == 'title')]
        if not title_row.empty:
            title = title_row[doc_col].iloc[0]
        
        doc_identifier = title if title and not pd.isna(title) else doc_col
        
        # Process each main category
        for main_category, subcategories in categories_to_extract.items():
            category_insights = {}
            
            for subcategory in subcategories:
                # Look for exact matches first
                subcategory_rows = df[df['Category'] == subcategory]
                
                # If not found, try partial matches
                if subcategory_rows.empty:
                    subcategory_rows = df[df['Category'].str.contains(subcategory, na=False)]
                
                # If still not found, look for it in SubCategory
                if subcategory_rows.empty and 'SubCategory' in df.columns:
                    subcategory_rows = df[df['SubCategory'] == subcategory]
                    
                    if subcategory_rows.empty:
                        subcategory_rows = df[df['SubCategory'].str.contains(subcategory, na=False)]
                
                if not subcategory_rows.empty:
                    subcategory_data = subcategory_rows[doc_col].dropna().tolist()
                    if subcategory_data:
                        category_insights[subcategory] = subcategory_data
            
            if category_insights:
                doc_insights[main_category] = category_insights
        
        insights[doc_identifier] = doc_insights
    
    return insights


def generate_insights_with_gpt4o(insights_data, api_key):
    """
    Pass the extracted research insights to GPT-4o and get concise bullet point insights.
    """
    if not insights_data:
        return ["No insights found in the filtered documents."]
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Format the structured insights data into a readable text format for the prompt
        formatted_insights = []
        
        for doc_id, doc_data in insights_data.items():
            formatted_insights.append(f"DOCUMENT: {doc_id}")
            
            for category, category_data in doc_data.items():
                formatted_insights.append(f"\n{category}:")
                
                for subcategory, values in category_data.items():
                    if isinstance(values, list):
                        formatted_insights.append(f"  - {subcategory}: {'; '.join(str(v) for v in values)}")
                    else:
                        formatted_insights.append(f"  - {subcategory}: {values}")
            
            formatted_insights.append("\n---\n")
        
        # Prepare the prompt with specific formatting instructions
        prompt = """
        You are an expert researcher analyzing e-cigarette and vaping studies. Below are detailed research insights from several studies, organized by document and category. 
        
        Based on these insights, generate 7-10 concise, insightful bullet points that capture the key findings, patterns, and implications across the studies.
        Focus on being specific, evidence-based, and highlighting both consensus and contradictions across studies.
        
        Pay special attention to:
        1. Key conclusions and novel findings
        2. Health outcomes and causal mechanisms 
        3. R&D implications
        4. Methodological strengths and limitations
        5. Areas of consensus vs. areas of contradiction

        IMPORTANT FORMATTING INSTRUCTION:
        - Use ONLY a single bullet point character '•' at the beginning of each insight
        - DO NOT use any secondary or nested bullet points
        - DO NOT start any line with any other bullet character or symbol
        
        Here are the research insights:
        
        {}
        
        Please respond with only the bullet points, each starting with a '•' character.
        """.format('\n'.join(formatted_insights))
        
        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise research insights with simple bullet points. Never use nested bullet points."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        # Extract and process the bullet points
        insights_text = response.choices[0].message.content
        
        # Split the text into bullet points, making sure each starts with •
        bullet_points = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line and line.startswith('•'):
                # Remove any potential nested bullets by replacing any bullet characters
                # that might appear after the initial bullet with their text equivalent
                clean_line = line.replace(' • ', ': ')  # Replace nested bullets with colons
                bullet_points.append(clean_line)
            elif line and bullet_points:  # For lines that might be continuation of previous bullet point
                # Make sure there are no bullet characters in continuation lines
                clean_line = line.replace('•', '')
                bullet_points[-1] += ' ' + clean_line
        
        # If no bullet points were found with •, try to parse by lines
        if not bullet_points:
            bullet_points = [line.strip().replace('•', '') for line in insights_text.split('\n') if line.strip()]
        
        return bullet_points
    
    except Exception as e:
        return [f"Error generating insights: {str(e)}"]
    

def display_gpt4o_insights(df, matching_docs):
    st.subheader("Research Insights")
    
    # Check if there are matching documents
    if not matching_docs:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        return
    
    # Get the API key from session state (already set in sidebar)
    api_key = st.session_state.openai_api_key
    
    # Create container with fixed height for scrollable content
    insights_container = st.container()
    
    with insights_container:
        # Always add the CSS for the insights wrapper
        st.markdown("""
        <style>
        .insights-wrapper {
            height: 525px;
            overflow-y: auto;
            padding: 0rem;
            border: 2px solid #f8d6d5;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if generate_button:  # Using the button from sidebar
            with st.spinner("Generating insights..."):
                # Extract research insights from matching documents
                research_insights = extract_research_insights_from_docs(df, matching_docs)
                
                if not research_insights:
                    st.markdown("<div class='insights-wrapper'>No research insights found in the filtered documents.</div>", unsafe_allow_html=True)
                    return
                
                # Generate insights using GPT-4o
                insights = generate_insights_with_gpt4o(research_insights, api_key)
                
                # Create the scrollable container with insights content
                insights_html = "<div class='insights-wrapper'>"
                for insight in insights:
                    insights_html += f"<p>{insight}</p>"
                insights_html += "</div>"
                
                st.markdown(insights_html, unsafe_allow_html=True)
            
            # Save the generated insights in session state for reuse
            st.session_state.generated_insights = insights
                        
        elif api_key and "generated_insights" in st.session_state:
            # Display previously generated insights if available
            insights_html = "<div class='insights-wrapper'>"
            for insight in st.session_state.generated_insights:
                insights_html += f"<p>{insight}</p>"
            insights_html += "</div>"
            
            st.markdown(insights_html, unsafe_allow_html=True)
        else:
            # For the empty state, we'll create a custom component that combines:
            # 1. The existing border styling
            # 2. A message about generating insights
            # 3. The wordcloud image loaded directly from file
            
            message = "Click the 'Generate Insights' button to analyze research findings."
            if not api_key:
                message = "Please enter your OpenAI API key to generate insights."
            
            try:
                # Load the wordcloud image directly from the Images folder
                wordcloud_path = "Images/ecigarette_research_wordcloud.png"
                import base64
                
                # Read and encode the image to base64
                with open(wordcloud_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Build HTML with embedded wordcloud image
                html = f"""
                <div class="insights-wrapper" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: left; margin-bottom: 0px; position: absolute; top: 8px; left: 20px; right: 0; z-index: 2;">{message}</p>
                    <img src="data:image/png;base64,{encoded_image}" style="width: 100%; height: 100%; object-fit: cover; padding: 35px 0px 15px 0px;" />
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                # Fallback if image loading fails
                st.markdown(f"""
                <div class="insights-wrapper" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: center;">{message}</p>
                    <p style="color: #999; font-size: 0.8em;">Unable to load wordcloud image: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                
# Load the Excel file
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('E_Cigarette_Research_Metadata_Consolidated.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Extract years from the dataframe - find rows where Category is 'publication_year'
def get_publication_years():
    if 'Category' in df.columns and 'publication_year' in df['Category'].values:
        # Get all rows where Category is 'publication_year'
        year_rows = df[df['Category'] == 'publication_year']
        # Extract years from all document columns (starting from column index 3)
        years = []
        doc_columns = df.columns[3:]
        for doc_col in doc_columns:
            year_values = year_rows[doc_col].dropna().astype(str)
            for year in year_values:
                try:
                    years.append(int(float(year)))
                except (ValueError, TypeError):
                    continue
        return years
    return [2011, 2025]  # Default range if data not found

# Get sample sizes
def get_sample_sizes():
    if 'Category' in df.columns and 'SubCategory' in df.columns:
        # Get all rows where SubCategory is 'total_size'
        size_rows = df[(df['SubCategory'] == 'total_size')]
        # Extract sizes from all document columns
        sizes = []
        doc_columns = df.columns[3:]
        for doc_col in doc_columns:
            size_values = size_rows[doc_col].dropna().astype(str)
            for size in size_values:
                try:
                    sizes.append(int(float(size)))
                except (ValueError, TypeError):
                    continue
        if sizes:
            min_size = min(sizes)
            # Set max_size to 10000 for the slider, but keep track of the actual max
            actual_max = max(sizes)
            return [min_size, min(10000, actual_max), actual_max]
    return [50, 10000, 15000]  # Default range if data not found

# Extract unique values for a given Category or SubCategory with their occurrence counts
def get_unique_values_filtered(category_name, subcategory_name=None, matching_docs=None):
    """
    Get unique values with occurrence counts based on filtered documents
    """
    value_counts = {}
    
    # If no matching docs provided, return just "All"
    if not matching_docs:
        return ["All"]
    
    # Handle different conditions based on what we're looking for
    if subcategory_name:
        # Looking for values in rows where SubCategory equals subcategory_name
        if 'SubCategory' in df.columns and subcategory_name in df['SubCategory'].values:
            rows = df[df['SubCategory'] == subcategory_name]
            
            # Extract values from matching document columns only
            for doc_col in matching_docs:
                col_values = rows[doc_col].dropna().astype(str)
                for value in col_values:
                    if value and value != "nan":
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
    else:
        # Looking for values in rows where Category equals category_name
        if 'Category' in df.columns and category_name in df['Category'].values:
            rows = df[df['Category'] == category_name]
            
            # Extract values from matching document columns only
            for doc_col in matching_docs:
                col_values = rows[doc_col].dropna().astype(str)
                for value in col_values:
                    if value and value != "nan":
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
    
    # Sort values by their occurrence count in decreasing order
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Format values with their counts in curly braces
    formatted_values = [f"{value} {{{count}}}" for value, count in sorted_values]
    
    # Add "All" as the first option
    return ["All"] + formatted_values


# Count documents that match the current filter criteria
def count_matching_documents(year_range, sample_size_range=None, publication_type=None, 
                            funding_source=None, study_design=None):
    # Start with all document columns
    doc_columns = df.columns[3:]
    matching_docs = []
    
    for doc_col in doc_columns:
        matches_all_criteria = True
        
        # Check year criteria
        if 'publication_year' in df['Category'].values:
            year_row = df[df['Category'] == 'publication_year']
            year_value = year_row[doc_col].iloc[0] if not year_row.empty else None
            
            if year_value:
                try:
                    year = int(float(year_value))
                    if year < year_range[0] or year > year_range[1]:
                        matches_all_criteria = False
                except (ValueError, TypeError):
                    matches_all_criteria = False
        
        # Check sample size criteria if enabled
        if sample_size_range and 'total_size' in df['SubCategory'].values:
            size_row = df[df['SubCategory'] == 'total_size']
            size_value = size_row[doc_col].iloc[0] if not size_row.empty else None
            
            if size_value:
                try:
                    size = int(float(size_value))
                    if size < sample_size_range[0] or size > sample_size_range[1]:
                        matches_all_criteria = False
                except (ValueError, TypeError):
                    matches_all_criteria = False
        
        
        # Check publication type criteria - handle values with counts in curly braces
        if publication_type and "All" not in publication_type and 'publication_type' in df['Category'].values:
            pub_row = df[df['Category'] == 'publication_type']
            pub_value = pub_row[doc_col].iloc[0] if not pub_row.empty else None
            
            if pub_value:
                # Extract just the value part before any curly braces for comparison
                pub_matches = False
                for selected_type in publication_type:
                    # Extract the base value without the count in curly braces
                    base_type = selected_type.split(' {')[0] if ' {' in selected_type else selected_type
                    if str(pub_value) == base_type:
                        pub_matches = True
                        break
                
                if not pub_matches:
                    matches_all_criteria = False
        
        # Check funding source criteria - handle values with counts in curly braces
        if funding_source and "All" not in funding_source and 'type' in df['SubCategory'].values:
            fund_row = df[df['SubCategory'] == 'type']
            fund_value = fund_row[doc_col].iloc[0] if not fund_row.empty else None
            
            if fund_value:
                # Extract just the value part before any curly braces for comparison
                fund_matches = False
                for selected_source in funding_source:
                    # Extract the base value without the count in curly braces
                    base_source = selected_source.split(' {')[0] if ' {' in selected_source else selected_source
                    if str(fund_value) == base_source:
                        fund_matches = True
                        break
                
                if not fund_matches:
                    matches_all_criteria = False
        
        # Check study design criteria - handle values with counts in curly braces
        if study_design and "All" not in study_design and 'primary_type' in df['SubCategory'].values:
            design_row = df[df['SubCategory'] == 'primary_type']
            design_value = design_row[doc_col].iloc[0] if not design_row.empty else None
            
            if design_value:
                # Extract just the value part before any curly braces for comparison
                design_matches = False
                for selected_design in study_design:
                    # Extract the base value without the count in curly braces
                    base_design = selected_design.split(' {')[0] if ' {' in selected_design else selected_design
                    if str(design_value) == base_design:
                        design_matches = True
                        break
                
                if not design_matches:
                    matches_all_criteria = False
        
        # If document matched all criteria, add to the list
        if matches_all_criteria:
            matching_docs.append(doc_col)
    
    return matching_docs

# Get filtered data for specific fields
def get_filtered_data(field_category, field_subcategory=None, matching_docs=None):
    if not matching_docs:
        return pd.DataFrame()
        
    if field_subcategory:
        rows = df[(df['Category'] == field_category) & (df['SubCategory'] == field_subcategory)]
    else:
        rows = df[df['Category'] == field_category]
    
    if rows.empty:
        return pd.DataFrame()
    
    # Extract data from matching document columns
    result_data = {}
    for doc_col in matching_docs:
        doc_name = doc_col  # Could use doc_col as the document name or extract a more readable name
        value = rows[doc_col].iloc[0] if not rows.empty else None
        if value and not pd.isna(value):
            result_data[doc_name] = value
    
    return pd.DataFrame({'document': list(result_data.keys()), 'value': list(result_data.values())})

# Generate publications by year chart data
def get_publications_by_year(matching_docs):
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

# Display logo
try:
    logo = Image.open("Images/IB-logo.png")
    st.image(logo, width=200)
except:
    st.write("Logo image not found.")

# Title
st.title("GenAI R&D Tool: E-Cigarette Research & Insights")

# Initialize session state for filters if they don't exist
if 'publication_type' not in st.session_state:
    st.session_state.publication_type = ["All"]
if 'funding_source' not in st.session_state:
    st.session_state.funding_source = ["All"]
if 'study_design' not in st.session_state:
    st.session_state.study_design = ["All"]
if 'year_range' not in st.session_state:
    # Get publication years
    years = get_publication_years()
    if years:
        min_year, max_year = min(years), max(years)
    else:
        min_year, max_year = 2011, 2025
    st.session_state.year_range = (min_year, max_year)
if 'enable_sample_size' not in st.session_state:
    st.session_state.enable_sample_size = False
if 'sample_size_filter' not in st.session_state:
    sample_size_range = get_sample_sizes()
    st.session_state.sample_size_filter = sample_size_range

# Define callback functions for each multiselect to handle the "All" selection logic
def on_publication_type_change():
    if "All" in st.session_state.publication_type_select and len(st.session_state.publication_type_select) > 1:
        if "All" not in st.session_state.publication_type:
            st.session_state.publication_type = ["All"]
        else:
            st.session_state.publication_type = [opt for opt in st.session_state.publication_type_select if opt != "All"]
    else:
        st.session_state.publication_type = st.session_state.publication_type_select
        
def on_funding_source_change():
    if "All" in st.session_state.funding_source_select and len(st.session_state.funding_source_select) > 1:
        if "All" not in st.session_state.funding_source:
            st.session_state.funding_source = ["All"]
        else:
            st.session_state.funding_source = [opt for opt in st.session_state.funding_source_select if opt != "All"]
    else:
        st.session_state.funding_source = st.session_state.funding_source_select
        
def on_study_design_change():
    if "All" in st.session_state.study_design_select and len(st.session_state.study_design_select) > 1:
        if "All" not in st.session_state.study_design:
            st.session_state.study_design = ["All"]
        else:
            st.session_state.study_design = [opt for opt in st.session_state.study_design_select if opt != "All"]
    else:
        st.session_state.study_design = st.session_state.study_design_select

def on_year_range_change():
    st.session_state.year_range = st.session_state.year_range_slider
    
# Update the on_sample_size_change function to handle the 10000+ case
def on_sample_size_change():
    sample_size_range = get_sample_sizes()
    actual_max = sample_size_range[2]
    
    # If the max slider value is 10000, set the actual filter to the true maximum
    if st.session_state.sample_size_slider[1] >= 10000:
        st.session_state.sample_size_filter = (st.session_state.sample_size_slider[0], actual_max)
    else:
        st.session_state.sample_size_filter = st.session_state.sample_size_slider

def on_enable_sample_size_change():
    st.session_state.enable_sample_size = st.session_state.enable_sample_size_checkbox


def generate_pyecharts_sunburst_data(df, matching_docs):
    """
    Generate hierarchical data structure for pyecharts sunburst chart
    from filtered matching_docs, showing top 5 from each hierarchy level:
    publication type, study design, and funding source
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
    """Create a pyecharts sunburst chart and return HTML"""
    
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
    """Create a pyecharts sunburst chart and return HTML"""
    
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
            series_name="E-Cigarette Research",
            data_pair=data,
            highlight_policy="ancestor",
            radius=[0, "95%"],
            sort_="null",
            levels=[
                {},  # Level 0 - Center: "Research"
                {    # Level 1 - Publication Types
                    "r0": "0",
                    "r": "35%",
                    "label": {"rotate": "0"}
                },
                {    # Level 2 - Study Designs 
                    "r0": "35%",
                    "r": "70%"
                },
                {    # Level 3 - Funding Sources
                    "r0": "70%",
                    "r": "95%",
                    "label": {
                        "rotate": "tangential",
                        "fontSize": 10
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
    """Function to generate and display the pyecharts sunburst in Streamlit"""
    # Generate the data
    sunburst_data = generate_pyecharts_sunburst_data(df, matching_docs)
    
    if not sunburst_data:
        st.warning("Not enough data to generate the chart. Please adjust your filters.")
        return
    
    # Create the HTML
    html_content = create_pyecharts_sunburst_html(sunburst_data)
    
    # Display in Streamlit
    st.components.v1.html(html_content, height=470, scrolling=False)


# Add a sidebar with filters
with st.sidebar:
    st.subheader("API Configuration")
    
    # Get OpenAI API key - in a production app, use st.secrets
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # Let user enter API key if not already set
    api_key = st.text_input(
        "Enter your OpenAI API key:",
        value=st.session_state.openai_api_key,
        type="password",
        key="api_key_input_sidebar"
    )
    st.session_state.openai_api_key = api_key
    
    # Custom CSS for the button with the Imperial Brands orange color
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #FF7417;
            color: white;
            font-weight: bold;
            border: 2px solid #FF0000;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #FFA664;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Generate Insights button in sidebar with the custom styling applied
    generate_button = st.button("Generate Insights") and api_key
    
    st.subheader("Filters")
    
    # Get publication years
    years = get_publication_years()
    if years:
        min_year, max_year = min(years), max(years)
    else:
        min_year, max_year = 2011, 2025
    
    # Year range slider
    year_range = st.slider(
        "Year Range", 
        min_value=min_year, 
        max_value=max_year, 
        value=st.session_state.year_range, 
        step=1,
        key="year_range_slider",
        on_change=on_year_range_change
    )
    
    # First, filter by year range to get initial matching documents
    initial_docs = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=["All"],
        funding_source=["All"],
        study_design=["All"]
    )
    
    
    # First filter: Publication Type with updated counts
    publication_types = get_unique_values_filtered(category_name="publication_type", 
                                                matching_docs=initial_docs)
    
    # Extract base values (without counts) from available options
    base_pub_types = ["All"] + [opt.split(" {")[0] for opt in publication_types if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_pub_types = [val.split(" {")[0] if " {" in val else val for val in st.session_state.publication_type]
    
    # Keep selections whose base values are in the available options
    valid_pub_types = []
    for i, base_val in enumerate(base_selected_pub_types):
        if base_val in base_pub_types or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_pub_types.append("All")
            else:
                # Find the option in publication_types that has the same base value
                matching_options = [opt for opt in publication_types if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_pub_types.append(matching_options[0])  # Use the option with updated count
    
    if not valid_pub_types:
        st.session_state.publication_type = ["All"]
    else:
        st.session_state.publication_type = valid_pub_types
    
    # Apply Publication Type filter
    st.multiselect(
        "Publication Type", 
        publication_types, 
        key="publication_type_select",
        default=st.session_state.publication_type,
        on_change=on_publication_type_change
    )
    
    # Filter docs after applying publication type
    docs_after_pub_type = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=st.session_state.publication_type,
        funding_source=["All"],
        study_design=["All"]
    )
    
    # Second filter: Funding Source with updated counts
    funding_sources = get_unique_values_filtered(category_name=None, subcategory_name="type", 
                                             matching_docs=docs_after_pub_type)
    
    # Extract base values (without counts) from available options
    base_funding_sources = ["All"] + [opt.split(" {")[0] for opt in funding_sources if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_funding_sources = [val.split(" {")[0] if " {" in val else val for val in st.session_state.funding_source]
    
    # Keep selections whose base values are in the available options
    valid_funding_sources = []
    for i, base_val in enumerate(base_selected_funding_sources):
        if base_val in base_funding_sources or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_funding_sources.append("All")
            else:
                # Find the option in funding_sources that has the same base value
                matching_options = [opt for opt in funding_sources if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_funding_sources.append(matching_options[0])  # Use the option with updated count
    
    if not valid_funding_sources:
        st.session_state.funding_source = ["All"]
    else:
        st.session_state.funding_source = valid_funding_sources
    
    # Apply Funding Source filter
    st.multiselect(
        "Funding Source", 
        funding_sources, 
        key="funding_source_select",
        default=st.session_state.funding_source,
        on_change=on_funding_source_change
    )
    
    # Filter docs after applying funding source
    docs_after_funding = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=st.session_state.publication_type,
        funding_source=st.session_state.funding_source,
        study_design=["All"]
    )
    
    # Third filter: Study Design with updated counts
    study_designs = get_unique_values_filtered(category_name=None, subcategory_name="primary_type", 
                                          matching_docs=docs_after_funding)
    
    # Extract base values (without counts) from available options
    base_study_designs = ["All"] + [opt.split(" {")[0] for opt in study_designs if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_study_designs = [val.split(" {")[0] if " {" in val else val for val in st.session_state.study_design]
    
    # Keep selections whose base values are in the available options
    valid_study_designs = []
    for i, base_val in enumerate(base_selected_study_designs):
        if base_val in base_study_designs or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_study_designs.append("All")
            else:
                # Find the option in study_designs that has the same base value
                matching_options = [opt for opt in study_designs if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_study_designs.append(matching_options[0])  # Use the option with updated count
    
    if not valid_study_designs:
        st.session_state.study_design = ["All"]
    else:
        st.session_state.study_design = valid_study_designs
    
    # Apply Study Design filter
    st.multiselect(
        "Study Design", 
        study_designs, 
        key="study_design_select",
        default=st.session_state.study_design,
        on_change=on_study_design_change
    )
    
    # Checkbox to enable/disable sample size range
    enable_sample_size = st.checkbox(
        "Enable Sample Size Filter", 
        value=st.session_state.enable_sample_size,
        key="enable_sample_size_checkbox",
        on_change=on_enable_sample_size_change
    )
    
    # Sample size range - only shown if checkbox is enabled
    if enable_sample_size:
        sample_size_range = get_sample_sizes()
        min_size = sample_size_range[0]
        slider_max = sample_size_range[1]  # This is either the actual max or 10000
        actual_max = sample_size_range[2]  # The true maximum value
        
        # Calculate the current slider values, respecting the 10000+ threshold
        current_min = st.session_state.sample_size_filter[0]
        current_max = st.session_state.sample_size_filter[1]
        
        # Set slider min/max values
        slider_min = current_min if current_min >= min_size else min_size
        adjusted_max = current_max
        if current_max > 10000:
            adjusted_max = 10000
        
        # Create the slider with custom formatting
        sample_size_values = st.slider(
            "Sample Size Range", 
            min_value=min_size, 
            max_value=slider_max,
            value=(slider_min, adjusted_max),
            key="sample_size_slider",
            on_change=on_sample_size_change,
            format="%d"  # Default format
        )
        
        # Custom label for the max value
        if sample_size_values[1] >= 10000:
            st.text(f"Selected range: {sample_size_values[0]} to 10000+")
            # Update the actual filter to include all values above 10000
            st.session_state.sample_size_filter = (sample_size_values[0], actual_max)
        else:
            # Normal case, just use the slider values
            st.session_state.sample_size_filter = sample_size_values
    else:
        sample_size_filter = None
        

# Apply filters and get matching documents
matching_docs = count_matching_documents(
    year_range=st.session_state.year_range,
    sample_size_range=st.session_state.sample_size_filter if st.session_state.enable_sample_size else None,
    publication_type=st.session_state.publication_type,
    funding_source=st.session_state.funding_source,
    study_design=st.session_state.study_design
)

# Display total number of documents selected in the sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader(f"Total Documents: {len(matching_docs)}")

# Tabs
tabs = st.tabs(["Overview", "Adverse Events", "Perceived Benefits", "Oral Health", "Research Trends", 
                "Contradictions & Conflicts", "Bias in Research", "Publication Level"])

# Overview Tab
with tabs[0]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Call the GPT-4o insights function
            display_gpt4o_insights(df, matching_docs)
            
        
        with col2:
            st.subheader("Publication Distribution")
            
            # Create radio buttons arranged horizontally for chart selection
            chart_type = st.radio(
                "Select Chart Type:",
                ["Overall", "Yearly", "Publication Type", "Funding Source", "Study Design"],
                horizontal=True
            )
            
            pub_df = get_publications_by_year(matching_docs)
            
            if not pub_df.empty:
                if chart_type == "Overall":
                    display_pyecharts_sunburst(df, matching_docs)
                    
                elif chart_type == "Yearly":
               
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
                
                elif chart_type == "Publication Type":
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
                
                elif chart_type == "Funding Source":
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
                
                elif chart_type == "Study Design":
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
            else:
                st.info("Publication year data not available for the filtered documents")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


with tabs[1]:  # Adverse Events tab
    st.subheader("Adverse Events")
    st.info("This section will display insights about adverse events reported in the research.")

with tabs[2]:  # Perceived Benefits tab
    st.subheader("Perceived Benefits")
    st.info("This section will display insights about perceived benefits reported in the research.")

with tabs[3]:  # Oral Health tab
    st.subheader("Oral Health")
    st.info("This section will display insights about oral health impacts reported in the research.")

with tabs[4]:  # Research Trends tab
    st.subheader("Research Trends")
    st.info("This section will display insights about research trends over time.")

with tabs[5]:  # Contradictions & Conflicts tab
    st.subheader("Contradictions & Conflicts")
    st.info("This section will highlight contradictory findings and conflicts in the research.")

with tabs[6]:  # Bias in Research tab
    st.subheader("Bias in Research")
    st.info("This section will analyze potential biases in the research methodologies and funding.")

with tabs[7]:  # Publication Level tab
    st.subheader("Publication Level")
    st.info("This section will provide metrics about publication impact and quality.")


# Show document details for debugging
if st.checkbox("Show Document Details"):
    st.subheader("Filtered Documents Details")
    
    if matching_docs:
        # Show titles for matching documents
        titles = []
        authors = []
        journals = []
        years = []
        
        title_rows = df[df['Category'] == 'title']
        author_rows = df[df['Category'] == 'authors']
        journal_rows = df[df['Category'] == 'journal']
        year_rows = df[df['Category'] == 'publication_year']
        
        for doc in matching_docs:
            title = title_rows[doc].iloc[0] if not title_rows.empty else "Unknown"
            author = author_rows[doc].iloc[0] if not author_rows.empty else "Unknown"
            journal = journal_rows[doc].iloc[0] if not journal_rows.empty else "Unknown"
            year = year_rows[doc].iloc[0] if not year_rows.empty else "Unknown"
            
            titles.append(title if not pd.isna(title) else "Unknown")
            authors.append(author if not pd.isna(author) else "Unknown")
            journals.append(journal if not pd.isna(journal) else "Unknown")
            years.append(year if not pd.isna(year) else "Unknown")
        
        doc_details = pd.DataFrame({
            'Document': matching_docs,
            'Title': titles,
            'Authors': authors,
            'Journal': journals,
            'Year': years
        })
        
        st.write(doc_details)
    else:
        st.write("No documents match the current filters")

# Show raw data if needed
if st.checkbox("Show Raw Data"):
    st.subheader("Sample Data")
    
    # Calculate the number of non-empty fields for each document
    doc_columns = df.columns[3:]  # Document columns start from index 3
    doc_completeness = {}
    
    for doc_col in doc_columns:
        # Count non-empty cells in this document column
        non_empty_count = df[doc_col].count()
        doc_completeness[doc_col] = non_empty_count
    
    # Sort documents by completeness (number of non-empty fields)
    sorted_docs = sorted(doc_completeness.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top 3 most complete documents
    top_3_docs = [doc[0] for doc in sorted_docs[:3]]
    
    # Display only the necessary columns: Main Category, Category, SubCategory, and the top 3 docs
    if top_3_docs:
        display_columns = ['Main Category', 'Category', 'SubCategory'] + top_3_docs
        st.write(df[display_columns])
        
    else:
        st.write("No document data available")
