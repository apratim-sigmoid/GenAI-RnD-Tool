import streamlit as st
import pandas as pd
import time
from functools import lru_cache
from openai import OpenAI



# Define the tab names for the progress tracking
tab_names = ["Overview", "Adverse Events", "Perceived Benefits", "Health Outcomes", 
             "Research Trends", "Contradictions & Conflicts", "Bias in Research", "Publication Level"]

def extract_research_insights_from_docs(df, matching_docs, categories_to_extract):
    """
    Extract comprehensive research insights from matching documents using custom categories.
    Only includes non-missing attributes to provide better context.
    
    Args:
        df (DataFrame): The dataframe containing all research data
        matching_docs (list): List of document columns that match filter criteria
        categories_to_extract (dict, optional): Dictionary of categories and subcategories to extract
                                               If None, uses default categories
    
    Returns:
        dict: Structured insights data organized by document and category
    """
    insights = {}
    
    # For each matching document, extract the insights
    for doc_col in matching_docs:
        doc_insights = {}
        
        # Get title if available
        title = None
        title_row = df[(df['Main Category'] == 'meta_data') & (df['Category'] == 'title')]
        if not title_row.empty:
            title = title_row[doc_col].iloc[0]
        if title is None:
            title_row = df[df['Category'] == 'title']
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
                    # Only include non-empty data
                    if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                        category_insights[subcategory] = subcategory_data
            
            # Only include categories with actual data
            if category_insights:
                doc_insights[main_category] = category_insights
        
        # Only include documents with actual insights
        if doc_insights:
            insights[doc_identifier] = doc_insights
    
    return insights


def generate_insights_with_gpt4o(insights_data, api_key, topic_name="Research", custom_focus_prompt=None):
    """
    Pass the extracted research insights to GPT-4o and get concise bullet point insights.
    
    Args:
        insights_data (dict): Structured insights data organized by document and category
        api_key (str): OpenAI API key
        topic_name (str): The name of the topic for prompt customization
        custom_focus_prompt (str, optional): Custom prompt section for specific focus areas
        
    Returns:
        tuple: (list of generated bullet points with insights, dict with token usage information)
    """
    if not insights_data:
        return [f"No {topic_name.lower()} insights found in the filtered documents."], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Format the structured insights data into a readable text format for the prompt with improved context
        formatted_insights = []
        
        for doc_id, doc_data in insights_data.items():
            formatted_insights.append(f"DOCUMENT: {doc_id}")
            
            for category, category_data in doc_data.items():
                # Add category header only if there's actual data
                if category_data:
                    formatted_insights.append(f"\n{category}:")
                    
                    for subcategory, values in category_data.items():
                        # Skip empty values
                        if not values or all(pd.isna(v) for v in values) or all(str(v).strip() == "" for v in values):
                            continue
                            
                        # Create a human-readable version of the subcategory by replacing dots and underscores
                        readable_subcategory = subcategory.replace('.', ' → ').replace('_', ' ').title()
                        
                        if isinstance(values, list):
                            # For lists, prefix each value with its meaning
                            if len(values) == 1:
                                formatted_insights.append(f"  - {readable_subcategory}: {values[0]}")
                            else:
                                formatted_insights.append(f"  - {readable_subcategory}:")
                                for i, val in enumerate(values):
                                    if str(val).strip():  # Only include non-empty values
                                        formatted_insights.append(f"      * Value {i+1}: {val}")
                        else:
                            if str(values).strip():  # Only include non-empty values
                                formatted_insights.append(f"  - {readable_subcategory}: {values}")
                        
            formatted_insights.append("\n---\n")
        
        # Prepare the prompt with specific formatting instructions
        prompt = f"""
        You are an expert researcher analyzing e-cigarette and vaping studies. Below are detailed {topic_name.lower()} insights from several studies, organized by document and category. 
        
        Based on these insights, generate 7-10 concise, insightful bullet points that capture the key findings, patterns, and implications across the studies.
        
        {custom_focus_prompt}
        
        IMPORTANT FORMATTING INSTRUCTION:
        - Use ONLY a single bullet point character '•' at the beginning of each insight
        - DO NOT use any secondary or nested bullet points
        - DO NOT start any line with any other bullet character or symbol
        
        Focus on precise measurements, numerical values, and specific technical details that directly enable product improvement.

        Here are the {topic_name.lower()} insights:
        
        {'\n'.join(formatted_insights)}
        
        Please respond with only the bullet points, each starting with a '•' character.
        """
        
        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that generates concise {topic_name.lower()} insights with simple bullet points. Never use nested bullet points. Always clearly indicate what metrics and units are being used."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        # Extract token usage information
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
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
        
        return bullet_points, token_usage
    
    except Exception as e:
        return [f"Error generating {topic_name.lower()} insights: {str(e)}"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    

# Add a cache for API responses
@lru_cache(maxsize=32)
def cached_generate_insights(insights_data_str, api_key, topic_name, custom_focus_prompt):
    """Cached version of the generate_insights function to avoid duplicate API calls"""
    # Convert insights_data_str back to dictionary
    import json
    insights_data = json.loads(insights_data_str)
    insights, token_usage = generate_insights_with_gpt4o(insights_data, api_key, topic_name, custom_focus_prompt)
    # Return both the insights and token usage
    return insights, token_usage


def display_insights(df, matching_docs, section_title="Research Insights", 
                     topic_name="Research", categories_to_extract=None, 
                     custom_focus_prompt=None,
                     wordcloud_path="Images/ecigarette_research_wordcloud.png",
                     enable_throttling=True,
                     tab_index=-1, height=525):
    """
    Modified function that handles multiple subtopics within a tab
    """
    st.subheader(section_title)
    
    # Check if there are matching documents
    if not matching_docs:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        return
    
    # Get the API key from session state
    api_key = st.session_state.openai_api_key
    
    # Create container for scrollable content
    insights_container = st.container()
    
    with insights_container:
        # Key for storing insights in session state - add support for subtopics
        insights_key = f"generated_{topic_name.lower().replace(' & ', '_').replace(' ', '_')}_insights"
        token_usage_key = f"{insights_key}_token_usage"
        
        # For health outcomes tab, use special handling for subtopics
        is_health_tab = (tab_index == 3)
        
        # Check if this tab should be processed
        is_current_tab = (st.session_state.insights_in_progress and 
                          tab_index == st.session_state.current_processing_tab and
                          tab_index not in st.session_state.completed_tabs)
        
        if is_current_tab:
            with st.spinner(f"Generating {topic_name.lower()} insights..."):
                try:
                    # Extract research insights from matching documents
                    research_insights = extract_research_insights_from_docs(df, matching_docs, categories_to_extract)
                    
                    if not research_insights:
                        insights = [f"No {topic_name.lower()} insights found in the filtered documents."]
                        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    else:
                        # Apply throttling if enabled
                        if enable_throttling and 'last_api_call' in st.session_state:
                            time_since_last_call = time.time() - st.session_state.last_api_call
                            if time_since_last_call < 1.0:  # Limit to 1 request per second
                                wait_time = 1.0 - time_since_last_call
                                time.sleep(wait_time)
                        
                        # Use the cached version to avoid duplicate API calls
                        import json
                        insights_data_str = json.dumps(research_insights)
                        insights, token_usage = cached_generate_insights(insights_data_str, api_key, topic_name, custom_focus_prompt)
                        
                        # Update the last API call timestamp
                        st.session_state.last_api_call = time.time()
                    
                    # Save the generated insights and token usage in session state
                    st.session_state[insights_key] = insights
                    st.session_state[token_usage_key] = token_usage
                    
                    # For health tab, we need to generate insights for all three health areas at once
                    if is_health_tab:
                        # Store insights for current subtopic area
                        if "Oral Health" in topic_name:
                            st.session_state["generated_oral_health_insights"] = insights
                            st.session_state["generated_oral_health_insights_token_usage"] = token_usage
                            
                            # Now generate insights for respiratory health
                            respiratory_prompt = """As an R&D specialist analyzing e-cigarette research on respiratory health, focus exclusively on specific product parameters and ingredients that impact respiratory health outcomes.
                                Avoid general statements about respiratory health and instead extract precise technical information that can directly improve product safety profiles.
                                
                                Emphasize:
                                1. Specific aerosol particle sizes from different device types and their measured deposition patterns
                                2. Exact chemical compounds at specific concentrations linked to respiratory irritation
                                3. Precise temperature/power settings associated with reduced respiratory effects (with numerical values)
                                4. Particular e-liquid formulations showing improved respiratory safety profiles (with measured data)
                                5. Specific device design elements that demonstrably filter or reduce harmful respiratory exposures
                                6. Comparative respiratory biomarker data between specific product designs (with exact measurements)
                                
                                Include specific quantitative data on aerosol physics, chemical composition, and physiological responses whenever available."""
                            
                            # Define categories specific to respiratory health
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
                                    "adverse_events.respiratory_events.breathing_difficulties.overall_percentage",
                                    "adverse_events.respiratory_events.breathing_difficulties.group_percentages",
                                    "adverse_events.respiratory_events.chest_pain.overall_percentage",
                                    "adverse_events.respiratory_events.chest_pain.group_percentages",
                                    "adverse_events.respiratory_events.other_respiratory_events",
                                    "adverse_events.oral_events.cough.overall_percentage",
                                    "adverse_events.oral_events.cough.time_course"
                                ],
                                "Causal Mechanisms": [
                                    "chemicals_implicated.name",
                                    "chemicals_implicated.effects",
                                    "biological_pathways.pathway",
                                    "biological_pathways.description"
                                ],
                                "Key Findings": [
                                    "main_conclusions", 
                                    "novel_findings"
                                ]
                            }
                            
                            respiratory_insights = extract_research_insights_from_docs(df, matching_docs, respiratory_categories)
                            if respiratory_insights:
                                respiratory_data_str = json.dumps(respiratory_insights)
                                respiratory_results, respiratory_token_usage = cached_generate_insights(respiratory_data_str, api_key, "Respiratory Health", respiratory_prompt)
                                st.session_state["generated_respiratory_health_insights"] = respiratory_results
                                st.session_state["generated_respiratory_health_insights_token_usage"] = respiratory_token_usage
                            else:
                                st.session_state["generated_respiratory_health_insights"] = ["No respiratory health insights found in the filtered documents."]
                                st.session_state["generated_respiratory_health_insights_token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                            
                            # Now generate insights for cardiovascular health
                            cardiovascular_prompt = """As an R&D specialist analyzing e-cigarette research on cardiovascular health, focus exclusively on specific product parameters and ingredients that impact cardiovascular health metrics.
                                Avoid general statements about cardiovascular risks and instead extract precise technical information that can directly inform product design and formulation.
                                
                                Emphasize:
                                1. Specific nicotine delivery patterns and their measured effects on heart rate/blood pressure (with exact values)
                                2. Exact chemical constituents linked to vascular effects with their concentrations
                                3. Precise operating parameters associated with minimized cardiovascular impact (with numerical data)
                                4. Particular device design elements showing improved cardiovascular safety profiles
                                5. Specific e-liquid formulations with measured reduced impact on endothelial function
                                6. Comparative cardiac biomarker data between specific product types (with exact measurements)
                                
                                Include exact measurements, concentration ranges, and physiological response data whenever available."""
                            
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
                                    "adverse_events.cardiovascular_events.heart_palpitation.overall_percentage",
                                    "adverse_events.cardiovascular_events.heart_palpitation.group_percentages",
                                    "adverse_events.cardiovascular_events.other_cardiovascular_events"
                                ],
                                "Causal Mechanisms": [
                                    "chemicals_implicated.name",
                                    "chemicals_implicated.effects",
                                    "biological_pathways.pathway",
                                    "biological_pathways.description"
                                ],
                                "Key Findings": [
                                    "main_conclusions", 
                                    "novel_findings"
                                ]
                            }
                            
                            cardiovascular_insights = extract_research_insights_from_docs(df, matching_docs, cardiovascular_categories)
                            if cardiovascular_insights:
                                cardiovascular_data_str = json.dumps(cardiovascular_insights)
                                cardiovascular_results, cardiovascular_token_usage = cached_generate_insights(cardiovascular_data_str, api_key, "Cardiovascular Health", cardiovascular_prompt)
                                st.session_state["generated_cardiovascular_health_insights"] = cardiovascular_results
                                st.session_state["generated_cardiovascular_health_insights_token_usage"] = cardiovascular_token_usage
                            else:
                                st.session_state["generated_cardiovascular_health_insights"] = ["No cardiovascular health insights found in the filtered documents."]
                                st.session_state["generated_cardiovascular_health_insights_token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    
                    # Display the insights we just generated with direct height styling
                    insights_html = f'<div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem;">'
                    for insight in insights:
                        insights_html += f"<p>{insight}</p>"
                    
                    # Add token usage information at the bottom
                    token_limit = 128000  # GPT-4o token limit
                    token_percentage = (token_usage["total_tokens"] / token_limit) * 100
                    insights_html += f"<p style='font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Tokens used: {token_usage['total_tokens']} ({token_percentage:.2f}% of {token_limit} limit)</p>"
                    
                    insights_html += "</div>"
                    st.markdown(insights_html, unsafe_allow_html=True)
                    
                    # Mark this tab as completed
                    st.session_state.completed_tabs.add(tab_index)
                    st.session_state.progress_status[tab_index] = "completed"
                    
                    # Setup for next tab
                    if tab_index < len(tab_names) - 1:
                        st.session_state.current_processing_tab = tab_index + 1
                        st.session_state.current_tab_index = tab_index + 1
                    else:
                        # All done
                        st.session_state.insights_in_progress = False
                        st.session_state.current_processing_tab = -1
                        st.session_state.current_tab_index = -1
                    
                    # Force a rerun to update UI and move to next tab
                    # Use a sleep to ensure the UI updates properly
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    # Continue with next tab even if there's an error
                    st.session_state.completed_tabs.add(tab_index)
                    if tab_index < len(tab_names) - 1:
                        st.session_state.current_processing_tab = tab_index + 1
                    else:
                        st.session_state.insights_in_progress = False
                        st.session_state.current_processing_tab = -1
                    time.sleep(0.5)
                    st.rerun()
                    
        elif insights_key in st.session_state:
            # Display previously generated insights with direct height styling
            insights_html = f'<div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem;">'
            for insight in st.session_state[insights_key]:
                insights_html += f"<p>{insight}</p>"
            
            # Add token usage information at the bottom if available
            if token_usage_key in st.session_state:
                token_usage = st.session_state[token_usage_key]
                token_limit = 128000  # GPT-4o token limit
                token_percentage = (token_usage["total_tokens"] / token_limit) * 100
                insights_html += f"<p style='font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Tokens used: {token_usage['total_tokens']} ({token_percentage:.1f}% of 128k limit)</p>"
            
            insights_html += "</div>"
            st.markdown(insights_html, unsafe_allow_html=True)
            
        else:
            # Empty state with wordcloud and direct height styling
            message = f"Click the 'Generate Insights' button to analyze {topic_name.lower()} findings."
            if not api_key:
                message = "Please enter your OpenAI API key to generate insights."
            
            try:
                # Load the wordcloud image
                import base64
                with open(wordcloud_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                html = f"""
                <div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: left; margin-bottom: 0px; position: absolute; top: 8px; left: 20px; right: 0; z-index: 2;">{message}</p>
                    <img src="data:image/png;base64,{encoded_image}" style="width: 100%; height: 100%; object-fit: cover; padding: 35px 0px 15px 0px;" />
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: center;">{message}</p>
                    <p style="color: #999; font-size: 0.8em;">Unable to load wordcloud image: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
