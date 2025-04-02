import streamlit as st
import pandas as pd
from PIL import Image
import requests

from insights_utils import extract_research_insights_from_docs, generate_insights_with_gpt4o, display_insights
from visualization_utils import display_publication_distribution


# Page config
st.set_page_config(
    page_title="IB GenAI R&D Tool",
    page_icon="🔬",
    layout="wide"
)


# Define the tab names for the progress tracking
tab_names = ["Overview", "Adverse Events", "Perceived Benefits", "Health Outcomes", 
             "Research Trends", "Contradictions & Conflicts", "Bias in Research", "Publication Level"]                
                
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
                
    st.markdown("""
    <style>
    /* Style for regular buttons - including all states */
    div.stButton > button:first-child {
        background-color: #5aac90;
        color: white;
        border: 2px solid transparent;  /* Start with transparent border */
    }
    
    div.stButton > button:hover {
        background-color: #5aac90;
        color: white;
        border: 2px solid orange;  /* Thicker red border on hover */
        box-sizing: border-box;  /* Ensure border doesn't change button size */
    }
    
    div.stButton > button:active, div.stButton > button:focus {
        background-color: #5aac90;
        color: yellow !important;
        border: 2px solid orange !important;  /* Thicker red border on active/focus */
        box-shadow: none;
    }
    
    /* Style for download buttons - including all states */
    div.stDownloadButton > button:first-child {
        background-color: #5aac90;
        color: white;
        border: 1px solid transparent;  /* Start with transparent border */
    }
    
    div.stDownloadButton > button:hover {
        background-color: #5aac90;
        color: white;
        border: 3px solid red;  /* Thicker red border on hover */
        box-sizing: border-box;  /* Ensure border doesn't change button size */
    }
    
    div.stDownloadButton > button:active, div.stDownloadButton > button:focus {
        background-color: #5aac90;
        color: white !important;
        border: 3px solid red !important;  /* Thicker red border on active/focus */
        box-shadow: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add these new session state variables
    if "insights_in_progress" not in st.session_state:
        st.session_state.insights_in_progress = False
    
    if "completed_tabs" not in st.session_state:
        st.session_state.completed_tabs = set()
    
    if "current_processing_tab" not in st.session_state:
        st.session_state.current_processing_tab = -1
    
    # Generate Insights button in sidebar with the custom styling applied
    generate_button = st.button("Generate Insights") and api_key
    
    # Initialize progress tracking in session state if not exists
    if "progress_status" not in st.session_state:
        st.session_state.progress_status = ["not_started"] * len(tab_names)
        
    if "current_tab_index" not in st.session_state:
        st.session_state.current_tab_index = -1  # -1 means not processing any tab
    
    # Display progress bar with custom styling
    if generate_button and not st.session_state.insights_in_progress:
        # Reset progress status when generate button is clicked
        st.session_state.insights_in_progress = True
        st.session_state.progress_status = ["waiting"] * len(tab_names)
        st.session_state.completed_tabs = set()
        st.session_state.current_processing_tab = 0  # Start with the first tab
        st.session_state.current_tab_index = 0  # Start with the first tab
        st.rerun()  # Trigger a rerun to start the process
        
    # Add custom CSS for the segmented progress bar
    st.markdown("""
    <style>
    .progress-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .progress-header {
        font-size: 0.9rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .segments-container {
        display: flex;
        height: 12px;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
    }
    .segment {
        flex: 1;
        height: 100%;
        margin: 0 1px;
    }
    .not-started {
        background-color: #ffebeb;  /* light pink for not started */
    }
    .waiting {
        background-color: #ffcccb;  /* Light red for waiting */
    }
    .processing {
        background-color: #FF7417;  /* Orange for processing (Imperial Brands color) */
        animation: pulse 1.5s infinite;
    }
    .completed {
        background-color: #5aac90;  /* Green for completed */
    }
    .segment-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 0.7rem;
    }
    .segment-label {
        flex: 1;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 0 1px;
    }
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the segmented progress bar
    st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
    
    # Create the segments container
    segments_html = "<div class='segments-container'>"
    for i in range(len(tab_names)):
        # Determine the status class
        if i in st.session_state.completed_tabs:
            status_class = "completed"
        elif i == st.session_state.current_processing_tab and st.session_state.insights_in_progress:
            status_class = "processing"
        elif i > st.session_state.current_processing_tab and st.session_state.insights_in_progress:
            status_class = "waiting"
        else:
            status_class = "not-started"
            
        # Add segment
        segments_html += f"<div class='segment {status_class}'></div>"
    
    segments_html += "</div>"
    
    st.markdown(segments_html, unsafe_allow_html=True)
    
    # Add segment labels
    # label_html = "<div class='segment-labels'>"
    # for tab in tab_names:
    #     label_html += f"<div class='segment-label'>{tab}</div>"
    # label_html += "</div>"
    # st.markdown(label_html, unsafe_allow_html=True)
    
    # Add a text indicator of what's being processed
    if st.session_state.insights_in_progress and st.session_state.current_processing_tab >= 0:
        current_tab = tab_names[st.session_state.current_processing_tab]
        st.markdown(f"<p style='text-align: center; margin-top: 0px; font-size: 0.8rem;'>Processing: {current_tab}</p>", unsafe_allow_html=True)
    elif st.session_state.completed_tabs and len(st.session_state.completed_tabs) == len(tab_names):
        st.markdown("<p style='text-align: center; margin-top: 0px; color: green; font-size: 0.8rem;'>✓ All insights generated!</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    
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
    st.subheader(f"Total Documents: {len(matching_docs)}")

# Tabs
tabs = st.tabs(["Overview", "Adverse Events", "Perceived Benefits", "Health Outcomes", "Research Trends", 
                "Contradictions & Conflicts", "Bias in Research", "Publication Level"])

# Overview Tab (Tab 0)
with tabs[0]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            overview_prompt = """Focus on being specific, evidence-based, and highlighting both consensus and contradictions across studies.
        
                Pay special attention to:
                1. Key conclusions and novel findings
                2. Health outcomes and causal mechanisms 
                3. R&D implications
                4. Methodological strengths and limitations
                5. Areas of consensus vs. areas of contradiction"""


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
            
            display_insights(
                df, 
                matching_docs,
                section_title="Research Insights",
                topic_name="Overall",
                categories_to_extract=categories_to_extract,
                custom_focus_prompt=overview_prompt,
                tab_index=0  # Add tab index
            )
        
        with col2:
            # Use the imported function to display visualizations
            display_publication_distribution(df, matching_docs)
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        

# Tab 1 (Adverse Events)
with tabs[1]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            adverse_events_prompt = """Focus on analyzing the adverse health effects of e-cigarettes with attention to severity, frequency, and biological mechanisms.
        
                Pay special attention to:
                1. Most commonly reported adverse events and their prevalence
                2. Severity and duration of adverse effects
                3. Specific chemicals or device factors linked to adverse outcomes
                4. Comparison of adverse events between different user groups and device types
                5. Evidence strength for causal relationships
                6. Clinical significance of reported adverse events"""


            # Define categories specific to adverse events
            adverse_events_categories = {
                "Health Outcomes": [
                    "respiratory_effects", "cardiovascular_effects", 
                    "neurological_effects", "psychiatric_effects", 
                    "other_health_outcomes"
                ],
                "Self-Reported Effects": [
                    "oral_events.sore_dry_mouth", "oral_events.cough", 
                    "respiratory_events.breathing_difficulties", "respiratory_events.chest_pain",
                    "neurological_events.headache", "neurological_events.dizziness",
                    "cardiovascular_events.heart_palpitation", "total_adverse_events"
                ],
                "Causal Mechanisms": [
                    "chemicals_implicated", "biological_pathways",
                    "device_factors", "usage_pattern_factors"
                ],
                "Key Findings": [
                    "main_conclusions", "limitations", "novel_findings"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Adverse Events Analysis",
                topic_name="Adverse Events",
                categories_to_extract=adverse_events_categories,
                custom_focus_prompt=adverse_events_prompt,
                tab_index=1  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for adverse events here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 2 (Perceived Benefits)
with tabs[2]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            perceived_benefits_prompt = """Focus on analyzing reported benefits of e-cigarette use with attention to objective measurements and subjective experiences.
        
                Pay special attention to:
                1. Most commonly reported health improvements and their prevalence
                2. Quality of life changes and their domains (physical, mental, social)
                3. Comparison of benefits between different user groups and device types
                4. Relative benefits compared to traditional cigarettes and other nicotine products
                5. Temporal patterns in perceived benefits (immediate vs. long-term)
                6. Correlation between user characteristics and reported benefits"""


            # Define categories specific to perceived benefits
            perceived_benefits_categories = {
                "Self-Reported Effects": [
                    "perceived_health_improvements.sensory.smell", 
                    "perceived_health_improvements.sensory.taste",
                    "perceived_health_improvements.physical.breathing",
                    "perceived_health_improvements.physical.physical_status",
                    "perceived_health_improvements.physical.stamina",
                    "perceived_health_improvements.mental.mood",
                    "perceived_health_improvements.mental.sleep_quality",
                    "perceived_health_improvements.quality_of_life"
                ],
                "Behavioral Patterns": [
                    "smoking_cessation.success_rates",
                    "smoking_cessation.comparison_to_other_methods",
                    "reasons_for_use"
                ],
                "R&D Insights": [
                    "comparative_benefits",
                    "consumer_experience_factors"
                ],
                "Key Findings": [
                    "main_conclusions", "novel_findings"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Perceived Benefits Analysis",
                topic_name="Perceived Benefits",
                categories_to_extract=perceived_benefits_categories,
                custom_focus_prompt=perceived_benefits_prompt,
                tab_index=2  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for perceived benefits here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 3 (Health Outcomes)
with tabs[3]:
    if matching_docs:
        # Initialize or get session state for selected health area
        if 'selected_health_area' not in st.session_state:
            st.session_state.selected_health_area = "oral"  # Default to oral health
        
        # Create keys for storing insights for each health area
        oral_insights_key = "generated_oral_health_insights"
        respiratory_insights_key = "generated_respiratory_health_insights"
        cardiovascular_insights_key = "generated_cardiovascular_health_insights"
        
        # Create a container for the entire tab with custom CSS for the anatomy diagram only
        st.markdown("""
        <style>
        /* Make the anatomy diagram larger */
        .anatomy-diagram {
            width: 100%;
            height: 450px;
        }
        
        /* Highlight boxes for different health sections */
        .highlight-box {
            border: 2px solid transparent;
            border-radius: 6px;
            padding: 4px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Functions to handle health area selection
        def select_oral_health():
            st.session_state.selected_health_area = "oral"
            
        def select_respiratory_health():
            st.session_state.selected_health_area = "lung"
            
        def select_cardiovascular_health():
            st.session_state.selected_health_area = "heart"
        
        # Create a row with two columns - one for content, one for anatomy
        col1, col2 = st.columns([1.5, 1])
        
        # Area for content
        with col1:
            # Button row for selecting health area - wrapped in a div with class for CSS targeting
            st.markdown('<div class="health-tab-buttons">', unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                oral_btn = st.button("Oral Health", 
                                    on_click=select_oral_health, 
                                    use_container_width=True,
                                    key="oral_health_btn")
                
            with btn_col2:
                respiratory_btn = st.button("Respiratory Health", 
                                          on_click=select_respiratory_health, 
                                          use_container_width=True,
                                          key="respiratory_health_btn")
                
            with btn_col3:
                cardiovascular_btn = st.button("Cardiovascular Health", 
                                             on_click=select_cardiovascular_health, 
                                             use_container_width=True,
                                             key="cardiovascular_health_btn")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Use JavaScript to style the active button
            active_area = st.session_state.selected_health_area
            if active_area == "oral":
                st.markdown("""
                <script>
                    document.querySelector('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]').classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            elif active_area == "lung":
                st.markdown("""
                <script>
                    document.querySelectorAll('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]')[1].classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            elif active_area == "heart":
                st.markdown("""
                <script>
                    document.querySelectorAll('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]')[2].classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            
            # Content based on selected health area
            if st.session_state.selected_health_area == "oral":
                oral_health_prompt = """Focus on analyzing the specific impacts of e-cigarettes on oral health with attention to clinical and self-reported outcomes.
        
                Pay special attention to:
                1. Effects on periodontal health, tooth structure, and oral mucosa
                2. Oral biomarkers and inflammatory responses
                3. Comparison with traditional smoking's oral health impacts
                4. Device and e-liquid factors that influence oral health outcomes
                5. Time-dependent changes in oral health metrics
                6. Clinical significance of observed oral health effects"""

                # Define categories specific to oral health
                oral_health_categories = {
                    "Health Outcomes": [
                        "oral_health.periodontal_health", 
                        "oral_health.caries_risk",
                        "oral_health.oral_mucosal_changes",
                        "oral_health.inflammatory_biomarkers",
                        "oral_health.other_oral_effects"
                    ],
                    "Self-Reported Effects": [
                        "adverse_events.oral_events.sore_dry_mouth",
                        "adverse_events.oral_events.mouth_tongue_sores",
                        "adverse_events.oral_events.gingivitis",
                        "adverse_events.oral_events.other_oral_events"
                    ],
                    "Causal Mechanisms": [
                        "chemicals_implicated", 
                        "biological_pathways"
                    ],
                    "Key Findings": [
                        "main_conclusions", 
                        "novel_findings"
                    ]
                }
                
                # Display insights for oral health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Oral Health Findings",
                    topic_name="Oral Health",
                    categories_to_extract=oral_health_categories,
                    custom_focus_prompt=oral_health_prompt,
                    tab_index=3,
                    height=430
                )
                
            elif st.session_state.selected_health_area == "lung":
                respiratory_prompt = """Focus on analyzing the specific impacts of e-cigarettes on respiratory health with attention to clinical outcomes, biomarkers, and patient experiences.
            
                Pay special attention to:
                1. Impacts on lung function and respiratory symptoms
                2. Specific respiratory conditions (asthma, COPD, wheezing)
                3. Biomarkers of respiratory inflammation or damage
                4. Comparison with traditional smoking's respiratory impacts
                5. Device and e-liquid factors influencing respiratory outcomes
                6. Time-dependent changes in respiratory health metrics"""

                # Categories specific to respiratory health
                respiratory_categories = {
                    "Health Outcomes": [
                        "respiratory_effects.measured_outcomes", 
                        "respiratory_effects.findings.description",
                        "respiratory_effects.findings.comparative_results",
                        "respiratory_effects.specific_conditions.asthma",
                        "respiratory_effects.specific_conditions.copd",
                        "respiratory_effects.specific_conditions.wheezing",
                        "respiratory_effects.specific_conditions.other_conditions",
                        "respiratory_effects.biomarkers",
                        "respiratory_effects.lung_function_tests.tests_performed",
                        "respiratory_effects.lung_function_tests.results"
                    ],
                    "Self-Reported Effects": [
                        "adverse_events.respiratory_events.breathing_difficulties",
                        "adverse_events.respiratory_events.chest_pain",
                        "adverse_events.respiratory_events.other_respiratory_events",
                        "adverse_events.oral_events.cough"
                    ],
                    "Causal Mechanisms": [
                        "chemicals_implicated", 
                        "biological_pathways"
                    ],
                    "Key Findings": [
                        "main_conclusions", 
                        "novel_findings"
                    ]
                }
                
                # Display insights for respiratory health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Respiratory Health Findings",
                    topic_name="Respiratory Health",
                    categories_to_extract=respiratory_categories,
                    custom_focus_prompt=respiratory_prompt,
                    tab_index=3,
                    height=430
                )
                
            elif st.session_state.selected_health_area == "heart":
                cardiovascular_prompt = """Focus on analyzing the specific impacts of e-cigarettes on cardiovascular health with attention to clinical measurements and physiological effects.
            
                Pay special attention to:
                1. Effects on heart rate, blood pressure, and vascular function
                2. Cardiovascular biomarkers and inflammatory responses
                3. Comparison with traditional smoking's cardiovascular impacts
                4. Device features and chemical constituents affecting cardiovascular outcomes
                5. Acute versus chronic cardiovascular effects
                6. Cardiovascular risk assessment for different user populations"""

                # Categories specific to cardiovascular health
                cardiovascular_categories = {
                    "Health Outcomes": [
                        "cardiovascular_effects.measured_outcomes", 
                        "cardiovascular_effects.findings.description",
                        "cardiovascular_effects.findings.comparative_results",
                        "cardiovascular_effects.blood_pressure",
                        "cardiovascular_effects.heart_rate",
                        "cardiovascular_effects.biomarkers"
                    ],
                    "Self-Reported Effects": [
                        "adverse_events.cardiovascular_events.heart_palpitation",
                        "adverse_events.cardiovascular_events.other_cardiovascular_events"
                    ],
                    "Causal Mechanisms": [
                        "chemicals_implicated", 
                        "biological_pathways"
                    ],
                    "Key Findings": [
                        "main_conclusions", 
                        "novel_findings"
                    ]
                }
                
                # Display insights for cardiovascular health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Cardiovascular Health Findings",
                    topic_name="Cardiovascular Health",
                    categories_to_extract=cardiovascular_categories,
                    custom_focus_prompt=cardiovascular_prompt,
                    tab_index=3,
                    height=430
                )
        
        # Anatomy diagram with server-side controlled highlighting
        with col2:
            # Fetch SVG content
            @st.cache_data
            def get_svg_content():
                """Fetch SVG content from echarts example"""
                url = "https://echarts.apache.org/examples/data/asset/geo/Veins_Medical_Diagram_clip_art.svg"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.text
                else:
                    st.error(f"Failed to fetch SVG: {response.status_code}")
                    return None
        
            # Get SVG content
            svg_content = get_svg_content()
        
            if svg_content:
                # Create HTML for the anatomy diagram with the current highlighted organ
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
                    <style>
                        html, body {{
                            margin: 0;
                            padding: 0;
                            width: 100%;
                            height: 100%;
                            overflow: hidden;
                        }}
                        #main {{
                            width: 100%;
                            height: 100%;
                        }}
                    </style>
                </head>
                <body>
                    <div id="main"></div>
                    <script>
                        // Initialize chart
                        var chartDom = document.getElementById('main');
                        var myChart = echarts.init(chartDom);
                        
                        // Register the SVG map
                        echarts.registerMap('organ_diagram', {{
                            svg: `{svg_content}`
                        }});
                        
                        var option = {{
                            tooltip: {{
                                formatter: function(params) {{
                                    switch(params.name) {{
                                        case 'lung':
                                            return 'Lungs - Respiratory System';
                                        case 'heart':
                                            return 'Heart - Cardiovascular System';
                                        case 'oral':
                                            return 'Oral Cavity - Oral Health';
                                        default:
                                            return params.name;
                                    }}
                                }}
                            }},
                            geo: {{
                                map: 'organ_diagram',
                                roam: false,
                                emphasis: {{
                                    focus: 'self',
                                    itemStyle: {{
                                        color: '#ff3333',  // Highlighting color
                                        borderWidth: 2,
                                        borderColor: '#ff0000',
                                        shadowBlur: 5,
                                        shadowColor: 'rgba(255, 0, 0, 0.5)'
                                    }}
                                }}
                            }}
                        }};
                        
                        myChart.setOption(option);
                        
                        // Set initial highlighting based on server-side state
                        const highlightedOrgan = "{st.session_state.selected_health_area}";
                        
                        if (highlightedOrgan && highlightedOrgan !== "None") {{
                            // First clear any existing highlights
                            myChart.dispatchAction({{
                                type: 'downplay',
                                geoIndex: 0
                            }});
                            
                            // Then highlight the requested organ
                            myChart.dispatchAction({{
                                type: 'highlight',
                                geoIndex: 0,
                                name: highlightedOrgan
                            }});
                        }}
                        
                        // Handle window resize
                        window.addEventListener('resize', function() {{
                            myChart.resize();
                        }});
                    </script>
                </body>
                </html>
                """
                        
                # Add a title for the anatomy visualization
                st.markdown("<h4 style='text-align: center; margin-top: -50px;'></h4>", unsafe_allow_html=True)
                
                # Display the anatomy diagram
                components_container = st.container()
                with components_container:
                    st.components.v1.html(html_template, height=650, scrolling=False)
                
            else:
                st.error("Could not load the anatomy diagram.")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        

# Tab 4 (Research Trends)
with tabs[4]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            research_trends_prompt = """Focus on analyzing how e-cigarette research has evolved over time and identifying emerging areas of focus.
        
                Pay special attention to:
                1. Shifts in research methodologies and study designs
                2. Evolution of research questions and hypotheses over time
                3. Emerging product trends and their research implications
                4. Gaps in current research and suggested future directions
                5. Geographic and institutional patterns in research focus
                6. Changes in user behavior patterns over time"""


            # Define categories specific to research trends
            research_trends_categories = {
                "Study Characteristics": [
                    "primary_type", "secondary_features", 
                    "time_periods", "data_collection_method"
                ],
                "Key Findings": [
                    "future_research_suggestions", "novel_findings"
                ],
                "Market Trends": [
                    "product_characteristics.device_evolution",
                    "product_characteristics.e_liquid_trends",
                    "product_characteristics.nicotine_concentration_trends",
                    "regulatory_impacts"
                ],
                "Behavioral Patterns": [
                    "usage_patterns.transitions",
                    "product_preferences"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Research Trends Analysis",
                topic_name="Research Trends",
                categories_to_extract=research_trends_categories,
                custom_focus_prompt=research_trends_prompt,
                tab_index=4  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for research trends here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 5 (Contradictions & Conflicts)
with tabs[5]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            contradictions_prompt = """Focus on identifying and analyzing contradictory findings, methodological disagreements, and areas of scientific controversy.
        
                Pay special attention to:
                1. Directly conflicting findings between studies
                2. Methodological differences that may explain contradictions
                3. Areas where evidence quality is particularly weak or inconsistent
                4. Funding sources and potential conflicts of interest
                5. Evolution of contradictions over time (resolving or intensifying)
                6. Impact of study design on observed outcomes"""


            # Define categories specific to contradictions and conflicts
            contradictions_categories = {
                "Key Findings": [
                    "contradictions.conflicts_with_literature",
                    "contradictions.internal_contradictions",
                    "generalizability",
                    "limitations"
                ],
                "Bias Assessment": [
                    "conflicts_of_interest",
                    "methodological_concerns",
                    "overall_quality_assessment"
                ],
                "Causal Mechanisms": [
                    "chemicals_implicated.evidence_strength",
                    "biological_pathways.evidence_strength"
                ],
                "R&D Insights": [
                    "comparative_benefits.vs_traditional_cigarettes.evidence_strength",
                    "harmful_ingredients.evidence_strength"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Contradictions & Conflicts Analysis",
                topic_name="Contradictions and Conflicts",
                categories_to_extract=contradictions_categories,
                custom_focus_prompt=contradictions_prompt,
                tab_index=5  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for contradictions and conflicts here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 6 (Bias in Research)
with tabs[6]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            bias_prompt = """Focus on analyzing potential sources of bias in e-cigarette research and their impact on findings.
        
                Pay special attention to:
                1. Patterns of funding influence on study outcomes
                2. Selection bias in study populations
                3. Measurement and reporting biases
                4. Handling of confounding variables
                5. Transparency in methods and conflicts disclosure
                6. Overall quality assessment and methodological rigor"""


            # Define categories specific to bias in research
            bias_categories = {
                "Bias Assessment": [
                    "selection_bias", "measurement_bias",
                    "confounding_factors", "attrition_bias",
                    "reporting_bias", "conflicts_of_interest",
                    "methodological_concerns", "overall_quality_assessment"
                ],
                "Meta Data": [
                    "funding_source.type",
                    "funding_source.specific_entities",
                    "funding_source.disclosure_statement"
                ],
                "Key Findings": [
                    "limitations", "generalizability"
                ],
                "Study Characteristics": [
                    "statistical_methods.adjustment_factors",
                    "methodology.control_variables"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Bias in Research Analysis",
                topic_name="Research Bias",
                categories_to_extract=bias_categories,
                custom_focus_prompt=bias_prompt,
                tab_index=6  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for bias in research here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 7 (Publication Level)
with tabs[7]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            publication_prompt = """Focus on analyzing publication patterns, citation impact, and research quality across e-cigarette literature.
        
                Pay special attention to:
                1. Publication trends across journals and publication types
                2. Geographic distribution of research
                3. Methodological quality across publication sources
                4. Relationship between publication metrics and funding sources
                5. Evolution of publication volume and focus over time
                6. Citation patterns and influential studies"""


            # Define categories specific to publication level
            publication_categories = {
                "Meta Data": [
                    "publication_type", "journal",
                    "citation_info", "publication_year",
                    "country_of_study", "authors"
                ],
                "Study Characteristics": [
                    "sample_characteristics.total_size",
                    "study_design.primary_type"
                ],
                "Key Findings": [
                    "statistical_summary.primary_outcomes",
                    "statistical_summary.secondary_outcomes",
                    "main_conclusions"
                ],
                "Bias Assessment": [
                    "overall_quality_assessment"
                ]
            }
            
            display_insights(
                df, 
                matching_docs,
                section_title="Publication Level Analysis",
                topic_name="Publication Metrics",
                categories_to_extract=publication_categories,
                custom_focus_prompt=publication_prompt,
                tab_index=7  # Add tab index
            )
            
        with col2:
            st.info("Add visualizations for publication metrics here")
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        

st.write("")

# Show document details for debugging
if st.checkbox("Show Document Details"):
    from data_display_utils import display_document_details
    display_document_details(df, matching_docs)

# Show raw data if needed
if st.checkbox("Show Raw Data"):
    from data_display_utils import display_raw_data
    display_raw_data(df)
