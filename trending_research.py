import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import re

def generate_comprehensive_paper_insights(df, doc, title, api_key):
    """
    Generate comprehensive R&D-focused insights for a specific e-cigarette research paper
    using all available categories from the dataset and emphasizing quantitative data
    relevant to product improvement. Only includes non-missing attributes.
    
    Args:
        df: The main dataframe containing research data
        doc: The document column for this specific paper
        title: The title of the paper
        api_key: OpenAI API key
    """
    
    # Define all categories to extract based on the comprehensive Excel structure
    comprehensive_categories = {
        "Key Findings": [
            "main_conclusions",
            "statistical_summary.primary_outcomes",
            "statistical_summary.secondary_outcomes",
            "novel_findings",
            "contradictions.conflicts_with_literature",
            "contradictions.internal_contradictions",
            "limitations",
            "generalizability",
            "future_research_suggestions"
        ],
        "Causal Mechanisms": [
            "chemicals_implicated.name",
            "chemicals_implicated.level_detected",
            "chemicals_implicated.effects",
            "chemicals_implicated.evidence_strength",
            "chemicals_implicated.mechanism",
            "device_factors.factor",
            "device_factors.effects",
            "device_factors.evidence_strength",
            "device_factors.mechanism",
            "usage_pattern_factors.pattern",
            "usage_pattern_factors.effects",
            "usage_pattern_factors.evidence_strength",
            "usage_pattern_factors.mechanism",
            "biological_pathways.pathway",
            "biological_pathways.description",
            "biological_pathways.evidence_strength"
        ],
        "R&D Insights": [
            "harmful_ingredients.name",
            "harmful_ingredients.health_impact",
            "harmful_ingredients.concentration_details",
            "harmful_ingredients.comparison_to_cigarettes",
            "harmful_ingredients.evidence_strength",
            "comparative_benefits.vs_traditional_cigarettes.benefit",
            "comparative_benefits.vs_traditional_cigarettes.magnitude",
            "comparative_benefits.vs_traditional_cigarettes.evidence_strength",
            "comparative_benefits.vs_other_nicotine_products",
            "device_design_implications.feature",
            "device_design_implications.impact",
            "device_design_implications.improvement_suggestion",
            "operating_parameters.temperature",
            "operating_parameters.wattage",
            "operating_parameters.puff_duration",
            "consumer_experience_factors.factor",
            "consumer_experience_factors.health_implication",
            "consumer_experience_factors.optimization_suggestion",
            "potential_innovation_areas.area",
            "potential_innovation_areas.current_gap",
            "potential_innovation_areas.potential_direction"
        ],
        "Device & Methodology": [
            "methodology.e_cigarette_specifications.device_types",
            "methodology.e_cigarette_specifications.generation",
            "methodology.e_cigarette_specifications.nicotine_content.concentrations",
            "methodology.e_cigarette_specifications.nicotine_content.delivery_method",
            "methodology.e_cigarette_specifications.e_liquid_types",
            "methodology.e_cigarette_specifications.flavors_studied",
            "methodology.e_cigarette_specifications.power_settings",
            "methodology.e_cigarette_specifications.heating_element",
            "methodology.e_cigarette_specifications.puff_parameters",
            "methodology.measurement_tools.technical_equipment",
            "methodology.measurement_tools.biological_measures"
        ],
        "Health Impacts": [
            "respiratory_effects.measured_outcomes",
            "respiratory_effects.findings.description",
            "respiratory_effects.findings.comparative_results",
            "respiratory_effects.biomarkers",
            "respiratory_effects.lung_function_tests.results",
            "cardiovascular_effects.measured_outcomes",
            "cardiovascular_effects.findings.description",
            "cardiovascular_effects.blood_pressure",
            "cardiovascular_effects.heart_rate",
            "cardiovascular_effects.biomarkers",
            "oral_health.periodontal_health.description",
            "oral_health.periodontal_health.measurements",
            "oral_health.inflammatory_biomarkers.description",
            "cancer_risk.description",
            "cancer_risk.biomarkers"
        ],
        "Consumer Experience": [
            "adverse_events.oral_events.sore_dry_mouth.overall_percentage",
            "adverse_events.oral_events.cough.overall_percentage",
            "adverse_events.respiratory_events.breathing_difficulties.overall_percentage",
            "adverse_events.total_adverse_events.overall_percentage",
            "perceived_health_improvements.sensory.smell.overall_percentage",
            "perceived_health_improvements.sensory.taste.overall_percentage",
            "perceived_health_improvements.physical.breathing.overall_percentage",
            "product_preferences.device_preferences.most_popular_devices",
            "product_preferences.flavor_preferences.most_popular_flavors",
            "product_preferences.nicotine_preferences.most_common_concentrations",
            "usage_patterns.frequency.daily_users_percentage",
            "usage_patterns.frequency.usage_sessions_per_day",
            "usage_patterns.intensity.average_puffs_per_session",
            "usage_patterns.nicotine_consumption.estimated_intake",
            "reasons_for_use.primary_reasons"
        ],
        "Market & Regulatory": [
            "product_characteristics.device_evolution",
            "product_characteristics.e_liquid_trends",
            "product_characteristics.nicotine_concentration_trends",
            "product_characteristics.price_trends",
            "consumer_behavior.purchasing_patterns",
            "consumer_behavior.brand_loyalty",
            "consumer_behavior.sales_channels",
            "regulatory_impacts.regulation_effects",
            "regulatory_impacts.policy_recommendations",
            "environmental_impact.waste_generation",
            "environmental_impact.pollution",
            "environmental_impact.sustainability_concerns"
        ]
    }
    
    # Extract research insights from the paper
    research_insights = {}
    doc_insights = {}
    
    # Process each main category
    for main_category, subcategories in comprehensive_categories.items():
        category_insights = {}
        
        for subcategory in subcategories:
            # Handle nested subcategories with dots
            if '.' in subcategory:
                parts = subcategory.split('.')
                
                # Try first with Category column
                found = False
                base_category = parts[0]
                sub_parts = '.'.join(parts[1:])
                
                # Look for rows where Category contains the base_category and SubCategory has the remaining parts
                category_rows = df[df['Category'].str.contains(base_category, na=False)]
                if not category_rows.empty:
                    sub_rows = category_rows[category_rows['SubCategory'].str.contains(sub_parts, na=False, regex=False)]
                    if not sub_rows.empty:
                        found = True
                        subcategory_data = sub_rows[doc].dropna().tolist()
                        # Only include non-empty data
                        if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                            category_insights[subcategory] = subcategory_data
                
                # If not found, try all combinations of column splits
                if not found:
                    # Try with direct SubCategory match
                    subcategory_rows = df[df['SubCategory'] == sub_parts]
                    if not subcategory_rows.empty:
                        subcategory_data = subcategory_rows[doc].dropna().tolist()
                        # Only include non-empty data
                        if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                            category_insights[subcategory] = subcategory_data
            else:
                # Direct match in Category
                category_rows = df[df['Category'] == subcategory]
                if not category_rows.empty:
                    subcategory_data = category_rows[doc].dropna().tolist()
                    # Only include non-empty data
                    if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                        category_insights[subcategory] = subcategory_data
                else:
                    # Try in SubCategory
                    subcategory_rows = df[df['SubCategory'] == subcategory]
                    if not subcategory_rows.empty:
                        subcategory_data = subcategory_rows[doc].dropna().tolist()
                        # Only include non-empty data
                        if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                            category_insights[subcategory] = subcategory_data
        
        # Only include categories with actual data
        if category_insights:
            doc_insights[main_category] = category_insights
    
    # Only include documents with actual insights
    if doc_insights:
        research_insights[title] = doc_insights
    
    if not research_insights:
        return ["No insights found for this paper."]
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Format the structured insights data with improved context preservation
        formatted_insights = []
        
        for doc_id, doc_data in research_insights.items():
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
        
        # Focused R&D prompt combining the best elements from all the prompts in IB_POC_Main.py
        rd_focused_prompt = f"""As an R&D specialist analyzing e-cigarette research, focus exclusively on extracting actionable, quantitative insights that can directly inform product improvements. 

            Your task is to analyze the research paper titled '{title}' and identify specific technical parameters, chemical formulations, and design elements that can enhance product safety and satisfaction.
            
            CRITICAL INSTRUCTIONS:
            1. Provide ONLY precise measurements, numerical values, and specific technical details - avoid general statements about e-cigarette safety.
            2. Focus on extracting exact chemical compounds/ingredients at specific concentrations that impact health outcomes (e.g., "formaldehyde at >40 μg/puff when device exceeds 240°C").
            3. Identify precise device parameters (temperature, wattage, coil material, puff duration) associated with reduced harmful outputs.
            4. Extract specific flavor compounds and quantitative data on their safety/satisfaction metrics.
            5. Highlight exact operating parameters that optimize nicotine delivery while minimizing harmful constituents.
            6. Provide numerical data on user satisfaction correlated with specific product characteristics.
            7. Include exact comparisons between device generations or design features with percentage improvements.
            8. Identify specific biological pathways and mechanisms of toxicity with measured values.
            
            DO NOT include general statements about e-cigarettes being harmful. Instead, provide specific actionable data points that can guide R&D efforts to improve product safety and satisfaction.
            """
                    
        # Prepare the prompt
        prompt = f"""
        You are an expert R&D specialist analyzing e-cigarette and vaping studies for a major e-cigarette manufacturer. Below are detailed research insights from a specific study, organized by category. 
        
        Based on these insights, generate 7-10 highly specific, quantitative, and actionable insights that can directly inform product development decisions.
        
        {rd_focused_prompt}

        IMPORTANT FORMATTING INSTRUCTION:
        - Use ONLY a single bullet point character '•' at the beginning of each insight
        - DO NOT use any secondary or nested bullet points
        - DO NOT start any line with any other bullet character or symbol
        - Focus on precise measurements, numerical values, and specific technical details
        - Always clarify what units or metrics are being used (%, °C, mg/mL, etc.)
        
        Here are the detailed research insights:
        
        {'\n'.join(formatted_insights)}
        
        Please respond with only the bullet points, each starting with a '•' character.
        """
        
        # Make API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a meticulous R&D specialist focused on extracting precise, quantitative data from research to improve e-cigarette products. You provide only specific technical details, exact measurements, and actionable recommendations based on research data. You always clearly indicate what metrics and units are being used."},
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
                # Remove any potential nested bullets
                clean_line = line.replace(' • ', ': ')
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


def display_trending_research(df, all_docs):
    """
    Display trending research feature highlighting new papers from 2024 and 2025
    with summaries, new harmful ingredients, and other relevant insights.
    
    Parameters:
    - df: The main dataframe containing the research data
    - all_docs: List of all document columns
    """
    
    st.markdown("""
    <style>
    .trending-header {
        background: linear-gradient(to bottom, #FF7417, #FFA866);
        color: white;
        padding: 5px 10px;  /* Reduced padding from 10px 15px */
        border-radius: 5px;
        margin-bottom: 10px;  /* Reduced from 15px */
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .trending-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #5aac90;
    }
    .trending-header h3 {
        margin: 0;  /* Remove default margins */
        font-size: 1.7rem;  /* Reduce font size from default */
    }
    .trending-card p {
        color: #666;
        font-size: 0.9rem;
    }
    .tag {
        display: inline-block;
        background-color: #f0f0f0;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .tag.new {
        background-color: #ffece0;
        color: #e64a19;
        border: 1px solid #e64a19;
    }
    .tag.harmful {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #c62828;
    }
    .tag.method {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #1565c0;
    }
    .tag.benefit {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #2e7d32;
    }
    .tag.health {
        background-color: #e1f5fe;
        color: #0277bd;
        border: 1px solid #0277bd;
    }
    .tag.behavioral {
        background-color: #f3e5f5;
        color: #7b1fa2;
        border: 1px solid #7b1fa2;
    }
    .tag.findings {
        background-color: #fff8e1;
        color: #ff8f00;
        border: 1px solid #ff8f00;
    }
    .tag.mechanism {
        background-color: #e0f2f1;
        color: #00796b;
        border: 1px solid #00796b;
    }
    .tag.innovation {
        background-color: #e8eaf6;
        color: #3f51b5;
        border: 1px solid #3f51b5;
    }
    .tag.technical {
        background-color: #ede7f6;
        color: #5e35b1;
        border: 1px solid #5e35b1;
    }
    .tag.environmental {
        background-color: #e0f7fa;
        color: #00acc1;
        border: 1px solid #00acc1;
    }
    .alert-badge {
        background-color: #c62828;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .research-metric {
        text-align: center;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #5aac90;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
    }
    .insights-container {
        height: 350px;
        overflow-y: auto;
        padding: 0.5rem;
        border: 2px solid #f8d6d5;
        border-radius: 0.5rem;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Filter for 2024 and 2025 papers (new papers)
    new_papers = get_papers_by_year(df, all_docs, 2025) + get_papers_by_year(df, all_docs, 2024)
    
    if len(new_papers) == 0:
        st.warning("No new research papers found in the dataset.")
        return
    
    # Header with alert count
    st.markdown(f"""
    <div class="trending-header">
        <h3><i class="fas fa-chart-line"></i> What's New in Research</h3>
        <div class="alert-badge">{len(new_papers)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="research-metric">
            <div class="metric-value">{len(new_papers)}</div>
            <div class="metric-label">NEW STUDIES</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Count new harmful ingredients
        new_harmful_ingredients = get_new_harmful_ingredients(df, all_docs, new_papers)
        st.markdown(f"""
        <div class="research-metric">
            <div class="metric-value">{len(new_harmful_ingredients)}</div>
            <div class="metric-label">NEW HARMFUL INGREDIENTS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get unique study designs
        study_designs = get_unique_values_for_papers(df, new_papers, "study_design", "primary_type")
        st.markdown(f"""
        <div class="research-metric">
            <div class="metric-value">{len(study_designs)}</div>
            <div class="metric-label">STUDY DESIGNS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Get funding types
        funding_types = get_unique_values_for_papers(df, new_papers, "funding_source", "type")
        st.markdown(f"""
        <div class="research-metric">
            <div class="metric-value">{len(funding_types)}</div>
            <div class="metric-label">FUNDING SOURCES</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different trending categories
    trending_tabs = st.tabs([
        "New Studies", 
        "Distribution",
        "Harmful Ingredients", 
        "Health Findings", 
        "Consumer Experience",
        "Comparative Analysis",
        "Regulatory & Policy",
        "Study Quality"
    ])
    
    with trending_tabs[0]:
        for i, doc in enumerate(new_papers):
            paper_details = get_paper_details(df, doc)
            
            if paper_details:
                pub_type = paper_details.get('publication_type', 'Research Paper')
                title = paper_details.get('title', f'New Research Paper {i+1}')
                authors = paper_details.get('authors', 'Various Authors')
                main_conclusions = paper_details.get('main_conclusions', '')
                pub_year = paper_details.get('publication_year', 'Recent')
                
                # Generate paper card
                st.markdown(f"""
                <div class="trending-card">
                    <h4>{title}</h4>
                    <p><strong>Authors:</strong> {authors}</p>
                    <p><strong>Type:</strong> <span class="tag method">{pub_type}</span></p>
                    <p><strong>Key Findings:</strong> {main_conclusions[:250]}{'...' if len(main_conclusions) > 250 else ''}</p>
                    <div>
                """, unsafe_allow_html=True)
                
                # Tags based on paper topics with actual publication year
                tags = generate_tags_for_paper(df, doc, pub_year)
                tag_html = ""
                for tag_text, tag_type in tags:
                    tag_html += f'<span class="tag {tag_type}">{tag_text}</span>'
                
                st.markdown(f"{tag_html}</div></div>", unsafe_allow_html=True)
                
                # Initialize paper-specific session state keys
                paper_key = f"paper_insights_{i}"
                if paper_key not in st.session_state:
                    st.session_state[paper_key] = False
                
                insights_key = f"paper_insights_data_{i}"
                if insights_key not in st.session_state:
                    st.session_state[insights_key] = []
                
                # Get API key
                api_key = st.session_state.get("openai_api_key", "")
                
                # Add generate insights button for each paper
                if st.button("Generate Insights For This Paper", key=f"insights_btn_{i}"):
                    if not api_key:
                        st.error("Please enter your OpenAI API key in the sidebar to generate insights.")
                    else:
                        st.session_state[paper_key] = True
                        with st.spinner("Generating insights..."):
                            # Generate insights using our comprehensive function
                            insights = generate_comprehensive_paper_insights(df, doc, title, api_key)
                            st.session_state[insights_key] = insights
                
                # Display insights if they exist
                if st.session_state[paper_key]:
                    st.subheader("Research Insights")
                    
                    # Create a container with custom styling for the insights
                    insights_html = '<div class="insights-container">'
                    for insight in st.session_state[insights_key]:
                        insights_html += f"<p>{insight}</p>"
                    insights_html += "</div>"
                    
                    st.markdown(insights_html, unsafe_allow_html=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
    
    
    # Distribution Tab
    with trending_tabs[1]:
        # Create two columns for side by side display
        col1, col2 = st.columns(2)
        
        with col1:
            # Paper types distribution
            pub_types = []
            for doc in new_papers:
                paper_details = get_paper_details(df, doc)
                if paper_details and 'publication_type' in paper_details:
                    pub_types.append(paper_details['publication_type'])
            
            if pub_types:
                pub_type_counts = pd.DataFrame(pd.Series(pub_types).value_counts()).reset_index()
                pub_type_counts.columns = ['Publication Type', 'Count']
                
                chart = alt.Chart(pub_type_counts).mark_bar().encode(
                    x=alt.X('Count:Q', title='Number of Papers', axis=alt.Axis(format='d')),
                    y=alt.Y('Publication Type:N', sort='-x', title=''),
                    color=alt.Color('Publication Type:N', legend=None, scale=alt.Scale(scheme='blueorange')),
                    tooltip=['Publication Type', 'Count']
                ).properties(
                    title='New Papers by Publication Type',
                    height=min(300, 50 * len(pub_type_counts))
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        with col2:
            # Funding source distribution
            funding_data = []
            for doc in new_papers:
                funding_type = get_value_for_paper(df, doc, "funding_source", "type")
                if funding_type:
                    funding_data.append(funding_type)
            
            if funding_data:
                funding_counts = pd.DataFrame(pd.Series(funding_data).value_counts()).reset_index()
                funding_counts.columns = ['Funding Source', 'Count']
                
                pie = alt.Chart(funding_counts).mark_arc().encode(
                    theta=alt.Theta('Count:Q'),
                    color=alt.Color('Funding Source:N', scale=alt.Scale(scheme='tableau10')),
                    tooltip=['Funding Source', 'Count']
                ).properties(
                    title='Funding Sources in New Papers',
                    height=300
                )
                
                st.altair_chart(pie, use_container_width=True)
                
 
    # Harmful Ingredients Tab
    with trending_tabs[2]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab identifies potentially harmful chemical compounds found in e-cigarettes according to recent research. It shows their health impacts, detected concentrations and the relative strength of evidence.</p>
        </div>
        """, unsafe_allow_html=True)
    
        if new_harmful_ingredients:
            # Create a dataframe for the ingredients
            ingredients_data = []
            for ingredient, details in new_harmful_ingredients.items():
                papers = details.get('papers', [])
                
                # Clean health impact text - remove numbering pattern
                health_impact = details.get('health_impact', 'Not specified')
                if health_impact and isinstance(health_impact, str):
                    # Remove numbering patterns like "1)", "1) ", "1. ", etc.
                    health_impact = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', health_impact.strip())
                
                # Clean evidence strength text - remove numbering pattern
                evidence = details.get('evidence_strength', 'Not specified')
                if evidence and isinstance(evidence, str):
                    # Remove numbering patterns
                    evidence = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', evidence.strip())
                
                # Clean ingredient name - remove numbering pattern
                clean_ingredient = ingredient
                if isinstance(ingredient, str):
                    clean_ingredient = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', ingredient.strip())
                
                ingredients_data.append({
                    'Ingredient': clean_ingredient,
                    'Papers': len(papers),
                    'Health Impact': health_impact,
                    'Evidence Strength': evidence
                })
            
            ingredients_df = pd.DataFrame(ingredients_data)
            
            # Reset index to start from 1 instead of 0
            ingredients_df.index = ingredients_df.index + 1
            
            # Display as a table
            st.dataframe(ingredients_df, use_container_width=True)
            
            # Display detailed information for each ingredient
            for ingredient, details in new_harmful_ingredients.items():
                # Clean ingredient name for display
                clean_ingredient = ingredient
                if isinstance(ingredient, str):
                    clean_ingredient = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', ingredient.strip())
                
                with st.expander(f"Details: {clean_ingredient}"):
                    # Clean and display health impact
                    health_impact = details.get('health_impact', 'Not specified')
                    if health_impact and isinstance(health_impact, str):
                        health_impact = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', health_impact.strip())
                    st.markdown(f"**Health Impact:** {health_impact}")
                    
                    # Clean and display evidence strength
                    evidence = details.get('evidence_strength', 'Not specified')
                    if evidence and isinstance(evidence, str):
                        evidence = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', evidence.strip())
                    st.markdown(f"**Evidence Strength:** {evidence}")
                    
                    # Clean and display comparison
                    comparison = details.get('comparison_to_cigarettes', 'Not specified')
                    if comparison and isinstance(comparison, str):
                        comparison = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', comparison.strip())
                    st.markdown(f"**Comparison to Traditional Cigarettes:** {comparison}")
                    
                    # Display paper titles (also clean them if necessary)
                    paper_titles = []
                    for title in details.get('paper_titles', ['Unknown']):
                        if isinstance(title, str):
                            clean_title = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', title.strip())
                            paper_titles.append(clean_title)
                        else:
                            paper_titles.append(title)
                            
                    st.markdown(f"**Found in Papers:** {', '.join(paper_titles)}")
        else:
            st.info("No new harmful ingredients identified in recently published papers.")
        
    
    # Health Findings Tab
    with trending_tabs[3]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab summarizes health effects from recent research across multiple body systems, including respiratory, cardiovascular, oral, neurological and other health impacts.</p>
        </div>
        """, unsafe_allow_html=True)
    
        health_categories = [
            "respiratory_effects", 
            "cardiovascular_effects", 
            "oral_health", 
            "neurological_effects",
            "psychiatric_effects",
            "cancer_risk",
            "developmental_effects"
        ]
        
        health_findings = get_health_findings(df, new_papers, health_categories)
        
        if health_findings:
            for category, findings in health_findings.items():
                with st.expander(f"{category.replace('_', ' ').title()} ({len(findings)} findings)"):
                    for finding in findings:
                        st.markdown(f"**Paper:** {finding.get('paper_title', 'Unknown paper')}")
                        st.markdown(f"**Finding:** {finding.get('description', '')}")
                        st.markdown("---")
        else:
            st.info("No specific health findings documented in recent papers.")
    
    
    # Consumer Experience Tab
    with trending_tabs[4]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab shows user preferences and experiences, including device preferences, flavor choices and perceived improvements in health or quality of life.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get product preferences
        preference_data = get_feature_data_for_papers(df, new_papers, "product_preferences", 
                                                    ["device_preferences.most_popular_devices", 
                                                     "flavor_preferences.most_popular_flavors",
                                                     "nicotine_preferences.most_common_concentrations"])
        
        if preference_data:
            with st.expander(f"Product Preferences ({len(preference_data)} findings)"):
                for item in preference_data:
                    st.markdown(f"**Popular Devices:** {item.get('device_preferences.most_popular_devices', 'Not specified')}")
                    st.markdown(f"**Popular Flavors:** {item.get('flavor_preferences.most_popular_flavors', 'Not specified')}")
                    st.markdown(f"**Common Nicotine Concentrations:** {item.get('nicotine_preferences.most_common_concentrations', 'Not specified')}")
                    st.markdown("---")
        
        # Get perceived health improvements
        health_improvement_data = get_feature_data_for_papers(df, new_papers, "perceived_health_improvements", 
                                                            ["sensory.smell.overall_percentage", 
                                                             "sensory.taste.overall_percentage",
                                                             "physical.breathing.overall_percentage"])
        
        if health_improvement_data:
            with st.expander(f"Perceived Health Improvements ({len(health_improvement_data)} findings)"):
                for item in health_improvement_data:
                    st.markdown(f"**Improved Smell (%):** {item.get('sensory.smell.overall_percentage', 'Not specified')}")
                    st.markdown(f"**Improved Taste (%):** {item.get('sensory.taste.overall_percentage', 'Not specified')}")
                    st.markdown(f"**Improved Breathing (%):** {item.get('physical.breathing.overall_percentage', 'Not specified')}")
                    st.markdown("---")
        
        # Get consumer experience factors
        experience_data = get_feature_data_for_papers(df, new_papers, "consumer_experience_factors", 
                                                    ["factor", "health_implication", "optimization_suggestion"])
        
        if experience_data:
            with st.expander(f"Consumer Experience Factors ({len(experience_data)} findings)"):
                for item in experience_data:
                    st.markdown(f"**Factor:** {item.get('factor', 'Not specified')}")
                    st.markdown(f"**Health Implication:** {item.get('health_implication', 'Not specified')}")
                    st.markdown(f"**Optimization Suggestion:** {item.get('optimization_suggestion', 'Not specified')}")
                    st.markdown("---")
        
        if not any([preference_data, health_improvement_data, experience_data]):
            st.info("No consumer experience data found in recent papers.")
    
    
    # Comparative Analysis Tab
    with trending_tabs[5]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab compares e-cigarettes to traditional cigarettes and other nicotine products, highlighting relative risks and benefits to help understand their comparative health impacts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get comparative benefits vs traditional cigarettes
        vs_trad_data = get_feature_data_for_papers(df, new_papers, "comparative_benefits", 
                                                 ["vs_traditional_cigarettes.benefit",
                                                  "vs_traditional_cigarettes.magnitude",
                                                  "vs_traditional_cigarettes.evidence_strength"])
        
        if vs_trad_data:
            with st.expander(f"Compared to Traditional Cigarettes ({len(vs_trad_data)} findings)"):
                for item in vs_trad_data:
                    st.markdown(f"**Benefit:** {item.get('vs_traditional_cigarettes.benefit', 'Not specified')}")
                    st.markdown(f"**Magnitude:** {item.get('vs_traditional_cigarettes.magnitude', 'Not specified')}")
                    st.markdown(f"**Evidence Strength:** {item.get('vs_traditional_cigarettes.evidence_strength', 'Not specified')}")
                    st.markdown("---")
        
        # Get comparative benefits vs other nicotine products
        vs_other_data = get_feature_data_for_papers(df, new_papers, "comparative_benefits", ["vs_other_nicotine_products"])
        
        if vs_other_data:
            with st.expander(f"Compared to Other Nicotine Products ({len(vs_other_data)} findings)"):
                for item in vs_other_data:
                    st.markdown(f"**Comparison:** {item.get('vs_other_nicotine_products', 'Not specified')}")
                    st.markdown("---")
        
        # Get harmful ingredients comparison to cigarettes
        harmful_comp_data = get_feature_data_for_papers(df, new_papers, "harmful_ingredients", ["comparison_to_cigarettes"])
        
        if harmful_comp_data:
            with st.expander(f"Harmful Ingredients Comparison ({len(harmful_comp_data)} findings)"):
                for item in harmful_comp_data:
                    st.markdown(f"**Comparison to Cigarettes:** {item.get('comparison_to_cigarettes', 'Not specified')}")
                    st.markdown("---")
        
        if not any([vs_trad_data, vs_other_data, harmful_comp_data]):
            st.info("No comparative analysis data found in recent papers.")
    
    
    # Regulatory & Policy Tab
    with trending_tabs[6]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab highlights regulatory implications and policy recommendations from recent research, showing how regulations affect the market and product development.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get regulatory impact data
        regulation_effects = get_value_for_papers(df, new_papers, "regulatory_impacts", "regulation_effects")
        policy_recommendations = get_value_for_papers(df, new_papers, "regulatory_impacts", "policy_recommendations")
        policy_relevance = get_value_for_papers(df, new_papers, "policy_relevance")
        specific_recommendations = get_value_for_papers(df, new_papers, "specific_recommendations")
        
        if regulation_effects:
            with st.expander("Regulation Effects"):
                for paper_title, value in regulation_effects.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if policy_recommendations:
            with st.expander("Policy Recommendations"):
                for paper_title, value in policy_recommendations.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if policy_relevance:
            with st.expander("Policy Relevance"):
                for paper_title, value in policy_relevance.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if specific_recommendations:
            with st.expander("Specific Recommendations"):
                for paper_title, value in specific_recommendations.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if not any([regulation_effects, policy_recommendations, policy_relevance, specific_recommendations]):
            st.info("No regulatory or policy data found in recent papers.")
    
    
    # Study Quality Tab
    with trending_tabs[7]:
        st.markdown("""
        <div style="background-color: #f8d6d5; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; color: #5a6268;"><i class="fas fa-info-circle"></i> This tab evaluates the quality and reliability of the new research, helping to assess the strength of evidence through bias assessment and methodological review.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get study quality data
        selection_bias = get_value_for_papers(df, new_papers, "selection_bias")
        measurement_bias = get_value_for_papers(df, new_papers, "measurement_bias")
        confounding = get_value_for_papers(df, new_papers, "confounding_factors")
        
        # Use the modified function with the word count filter for conflicts of interest
        # This will exclude entries with 3 or fewer words
        conflicts = get_value_for_papers(df, new_papers, "conflicts_of_interest", "description", min_word_count=3)
        
        overall_quality = get_value_for_papers(df, new_papers, "overall_quality_assessment")
        limitations = get_value_for_papers(df, new_papers, "limitations")
        
        if overall_quality:
            with st.expander("Overall Quality Assessment"):
                for paper_title, value in overall_quality.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if limitations:
            with st.expander("Study Limitations"):
                for paper_title, value in limitations.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if conflicts:
            with st.expander("Conflicts of Interest"):
                for paper_title, value in conflicts.items():
                    st.markdown(f"**{paper_title}:** {value}")
                    st.markdown("---")
        
        if selection_bias or measurement_bias or confounding:
            with st.expander("Bias Assessment"):
                if selection_bias:
                    st.subheader("Selection Bias")
                    for paper_title, value in selection_bias.items():
                        st.markdown(f"**{paper_title}:** {value}")
                        st.markdown("---")
                
                if measurement_bias:
                    st.subheader("Measurement Bias")
                    for paper_title, value in measurement_bias.items():
                        st.markdown(f"**{paper_title}:** {value}")
                        st.markdown("---")
                
                if confounding:
                    st.subheader("Confounding Factors")
                    for paper_title, value in confounding.items():
                        st.markdown(f"**{paper_title}:** {value}")
                        st.markdown("---")
        
        if not any([selection_bias, measurement_bias, confounding, conflicts, overall_quality, limitations]):
            st.info("No study quality assessment data found in recent papers.")
        

# Add these helper functions to your code to support the new tabs:

def get_feature_data_for_papers(df, papers, category, subcategories):
    """
    Extract specific feature data for all papers based on category and subcategories
    
    Parameters:
    - df: The main dataframe
    - papers: List of paper column names
    - category: The main category to look for
    - subcategories: List of subcategories to extract
    
    Returns:
    - List of dictionaries with feature data for each paper
    """
    all_data = []
    
    for doc in papers:
        # Get paper title for reference
        title_row = df[df['Category'] == 'title']
        paper_title = title_row[doc].iloc[0] if not title_row.empty else "Unknown paper"
        
        paper_data = {
            'paper': doc,
            'paper_title': paper_title
        }
        
        has_data = False
        
        for subcategory in subcategories:
            # First look in SubCategory
            if subcategory in df['SubCategory'].values:
                rows = df[df['SubCategory'] == subcategory]
                if not rows.empty and doc in rows.columns:
                    value = rows[doc].iloc[0]
                    if value and not pd.isna(value):
                        # Clean numbering pattern if present
                        if isinstance(value, str):
                            value = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', value.strip())
                        paper_data[subcategory] = value
                        has_data = True
            
            # Some values might be in Category instead
            elif subcategory in df['Category'].values:
                rows = df[df['Category'] == subcategory]
                if not rows.empty and doc in rows.columns:
                    value = rows[doc].iloc[0]
                    if value and not pd.isna(value):
                        # Clean numbering pattern if present
                        if isinstance(value, str):
                            value = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', value.strip())
                        paper_data[subcategory] = value
                        has_data = True
            
            # For nested subcategories with dots
            elif '.' in subcategory:
                parts = subcategory.split('.')
                base_category = parts[0]
                sub_parts = '.'.join(parts[1:])
                
                # Look for rows where Category contains the base_category
                category_rows = df[df['Category'].str.contains(base_category, na=False)]
                if not category_rows.empty:
                    sub_rows = category_rows[category_rows['SubCategory'].str.contains(sub_parts, na=False, regex=False)]
                    if not sub_rows.empty and doc in sub_rows.columns:
                        value = sub_rows[doc].iloc[0]
                        if value and not pd.isna(value):
                            # Clean numbering pattern if present
                            if isinstance(value, str):
                                value = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', value.strip())
                            paper_data[subcategory] = value
                            has_data = True
                
                # Try one more approach for nested categories
                if subcategory not in paper_data:
                    # Try to match the category containing the base category
                    cat_match = df[df['Category'].str.contains(base_category, na=False, regex=False)]
                    if not cat_match.empty:
                        # And subcategory containing the subparts
                        subcat_match = cat_match[cat_match['SubCategory'].str.contains(sub_parts, na=False, regex=False)]
                        if not subcat_match.empty and doc in subcat_match.columns:
                            value = subcat_match[doc].iloc[0]
                            if value and not pd.isna(value):
                                if isinstance(value, str):
                                    value = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', value.strip())
                                paper_data[subcategory] = value
                                has_data = True
        
        # Only add this paper if it has at least one data point
        if has_data:
            all_data.append(paper_data)
    
    return all_data


def get_value_for_papers(df, papers, category, subcategory=None, min_word_count=None):
    """
    Get values for a specified category/subcategory for all papers
    
    Parameters:
    - df: The main dataframe
    - papers: List of paper column names
    - category: The category to look for
    - subcategory: Optional subcategory
    - min_word_count: Optional minimum word count filter (when not None, excludes values with fewer words)
    
    Returns:
    - Dictionary with paper titles as keys and values found
    """
    results = {}
    
    for doc in papers:
        # Get paper title for reference
        title_row = df[df['Category'] == 'title']
        paper_title = title_row[doc].iloc[0] if not title_row.empty else f"Paper {doc}"
        
        # Look for the value based on whether subcategory is provided
        if subcategory:
            # First try exact match on both category and subcategory
            rows = df[(df['Category'] == category) & (df['SubCategory'] == subcategory)]
            
            # If no exact match, try with contains
            if rows.empty:
                rows = df[(df['Category'].str.contains(category, na=False)) & 
                         (df['SubCategory'] == subcategory)]
            
            # If still no match, try with contains for both
            if rows.empty:
                rows = df[(df['Category'].str.contains(category, na=False)) & 
                         (df['SubCategory'].str.contains(subcategory, na=False))]
        else:
            # Look directly in Category with exact match
            rows = df[df['Category'] == category]
            
            # If no exact match, try with contains
            if rows.empty:
                rows = df[df['Category'].str.contains(category, na=False)]
        
        # If we found matching rows, get the value
        if not rows.empty and doc in rows.columns:
            value = rows[doc].iloc[0]
            if value and not pd.isna(value):
                # Clean numbering pattern if present
                if isinstance(value, str):
                    value = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', value.strip())
                    
                    # Apply minimum word count filter if specified
                    if min_word_count is not None:
                        word_count = len(value.split())
                        if word_count <= min_word_count:
                            # Skip this value as it has too few words
                            continue
                
                results[paper_title] = value
    
    return results

                            
def get_papers_by_year(df, all_docs, year):
    """Get document columns for papers published in the specified year"""
    matching_papers = []
    
    for doc_col in all_docs:
        # Find the year for this document
        if 'publication_year' in df['Category'].values:
            year_row = df[df['Category'] == 'publication_year']
            if not year_row.empty:
                year_value = year_row[doc_col].iloc[0]
                if year_value and not pd.isna(year_value):
                    try:
                        doc_year = int(float(year_value))
                        if doc_year == year:
                            matching_papers.append(doc_col)
                    except (ValueError, TypeError):
                        pass
    
    return matching_papers


def get_paper_details(df, doc_col):
    """Extract key details about a paper from the dataframe"""
    paper_details = {}
    
    # Extract common metadata fields
    metadata_fields = {
        'title': ('Category', 'title'),
        'authors': ('Category', 'authors'),
        'journal': ('Category', 'journal'),
        'publication_year': ('Category', 'publication_year'),
        'doi': ('Category', 'doi'),
        'publication_type': ('Category', 'publication_type'),
        'country_of_study': ('Category', 'country_of_study'),
        'main_conclusions': ('Category', 'main_conclusions')
    }
    
    for field, (cat_type, cat_value) in metadata_fields.items():
        if cat_type == 'Category':
            rows = df[df['Category'] == cat_value]
        else:
            rows = df[df['SubCategory'] == cat_value]
        
        if not rows.empty:
            value = rows[doc_col].iloc[0]
            if value and not pd.isna(value):
                paper_details[field] = value
    
    return paper_details


def get_unique_values_for_papers(df, papers, category_field, subcategory_field=None):
    """Get unique values for a specified field across the selected papers"""
    unique_values = set()
    
    for doc in papers:
        if subcategory_field:
            rows = df[df['SubCategory'] == subcategory_field]
        else:
            rows = df[df['Category'] == category_field]
            
        if not rows.empty:
            value = rows[doc].iloc[0]
            if value and not pd.isna(value):
                unique_values.add(value)
    
    return list(unique_values)


def get_value_for_paper(df, doc, category_field, subcategory_field=None):
    """Get value for a specified field for a single paper"""
    if subcategory_field:
        rows = df[df['SubCategory'] == subcategory_field]
    else:
        rows = df[df['Category'] == category_field]
        
    if not rows.empty:
        value = rows[doc].iloc[0]
        if value and not pd.isna(value):
            return value
    
    return None


def get_new_harmful_ingredients(df, all_docs, new_papers):
    """
    Identify harmful ingredients that appear in 2025 papers but not in earlier papers
    Returns a dictionary of ingredients with their details
    """
    # Get all harmful ingredients from 2025 papers
    new_ingredients = {}
    
    # First find rows related to harmful ingredients
    ingredient_rows = df[df['SubCategory'] == 'name']
    impact_rows = df[df['SubCategory'] == 'health_impact']
    evidence_rows = df[df['SubCategory'] == 'evidence_strength']
    comparison_rows = df[df['SubCategory'] == 'comparison_to_cigarettes']
    
    # Get ingredients from new papers
    for doc in new_papers:
        if not ingredient_rows.empty:
            ingredient = ingredient_rows[doc].iloc[0]
            if ingredient and not pd.isna(ingredient):
                # Get paper title for reference
                title_row = df[df['Category'] == 'title']
                paper_title = title_row[doc].iloc[0] if not title_row.empty else "Unknown paper"
                
                # Get additional details if available
                health_impact = impact_rows[doc].iloc[0] if not impact_rows.empty else None
                evidence_strength = evidence_rows[doc].iloc[0] if not evidence_rows.empty else None
                comparison = comparison_rows[doc].iloc[0] if not comparison_rows.empty else None
                
                # Add or update ingredient info
                if ingredient not in new_ingredients:
                    new_ingredients[ingredient] = {
                        'papers': [doc],
                        'paper_titles': [paper_title],
                        'health_impact': health_impact,
                        'evidence_strength': evidence_strength,
                        'comparison_to_cigarettes': comparison
                    }
                else:
                    new_ingredients[ingredient]['papers'].append(doc)
                    new_ingredients[ingredient]['paper_titles'].append(paper_title)
    
    # Now check if these ingredients appear in older papers
    old_papers = [doc for doc in all_docs if doc not in new_papers]
    old_ingredients = set()
    
    for doc in old_papers:
        if not ingredient_rows.empty:
            ingredient = ingredient_rows[doc].iloc[0]
            if ingredient and not pd.isna(ingredient):
                old_ingredients.add(ingredient)
    
    # Filter to keep only ingredients that are new in 2025
    truly_new_ingredients = {k: v for k, v in new_ingredients.items() if k not in old_ingredients}
    
    return truly_new_ingredients


def get_health_findings(df, papers, health_categories):
    """Extract health findings from the papers for each category"""
    findings = {}
    
    for category in health_categories:
        category_findings = []
        
        # Look for description fields within each category
        description_rows = df[df['SubCategory'].str.contains('description', na=False) & 
                             df['Category'].str.contains(category, na=False)]
        
        for doc in papers:
            for _, row in description_rows.iterrows():
                finding = row[doc]
                if finding and not pd.isna(finding):
                    # Get paper title
                    title_row = df[df['Category'] == 'title']
                    paper_title = title_row[doc].iloc[0] if not title_row.empty else "Unknown paper"
                    
                    category_findings.append({
                        'paper': doc,
                        'paper_title': paper_title,
                        'description': finding
                    })
        
        if category_findings:
            findings[category] = category_findings
    
    return findings


def generate_tags_for_paper(df, doc, pub_year=None):
    """Generate relevant tags for a paper based on its content and publication year"""
    tags_by_type = {
        'year': [],
        'study_design': [],
        'harmful': [],
        'other': []
    }
    used_values = set()  # Track values we've already added to avoid duplicates
    
    # Get paper title to check for duplicates
    paper_title = None
    title_row = df[df['Category'] == 'title']
    if not title_row.empty:
        paper_title = title_row[doc].iloc[0]
    
    # Add publication year tag first (instead of fixed "2025")
    if pub_year and not pd.isna(pub_year):
        try:
            # Try to format as integer year if possible
            year_tag = str(int(float(pub_year)))
            tags_by_type['year'].append((year_tag, "new"))
        except (ValueError, TypeError):
            # If not a valid number, use as is
            tags_by_type['year'].append((str(pub_year), "new"))
    else:
        # Default tag if no year is available
        tags_by_type['year'].append(("New", "new"))
    
    # Check for various categories of interest
    tag_checks = [
        # Study design - priority 2
        ('study_design', 'primary_type', 'method', 'study_design'),
        
        # Harmful ingredients - priority 3
        ('harmful_ingredients', 'name', 'harmful', 'harmful'),
        
        # Current tags - other categories
        ('device_design_implications', 'feature', 'method', 'other'),
        ('comparative_benefits', 'vs_traditional_cigarettes.benefit', 'benefit', 'other'),
        ('respiratory_effects', 'measured_outcomes', 'method', 'other'),
        ('e_cigarette_specifications', 'device_types', 'method', 'other'),
        
        # Additional health outcome tags
        ('cardiovascular_effects', 'measured_outcomes', 'health', 'other'),
        ('cancer_risk', 'description', 'health', 'other'),
        ('oral_health', 'periodontal_health.description', 'health', 'other'),
        ('neurological_effects', 'specific_outcomes', 'health', 'other'),
        
        # Behavioral pattern tags
        ('reasons_for_use', 'primary_reasons', 'behavioral', 'other'),
        ('smoking_cessation', 'success_rates', 'behavioral', 'other'),
        ('product_preferences', 'flavor_preferences.most_popular_flavors', 'behavioral', 'other'),
        
        # R&D and innovation tags
        ('potential_innovation_areas', 'area', 'innovation', 'other'),
        ('operating_parameters', 'temperature', 'technical', 'other'),
        
        # Key findings tags
        ('novel_findings', '-', 'findings', 'other'),
        ('limitations', '-', 'findings', 'other'),
        
        # Biological mechanisms
        ('biological_pathways', 'pathway', 'mechanism', 'other'),
        
        # Environmental impact
        ('waste_generation', '-', 'environmental', 'other'),
        ('pollution', '-', 'environmental', 'other')
    ]
    
    for category, subcategory, tag_type, priority in tag_checks:
        # First check if subcategory directly exists in SubCategory
        if subcategory in df['SubCategory'].values:
            rows = df[df['SubCategory'] == subcategory]
            if not rows.empty:
                value = rows[doc].iloc[0]
                if value and not pd.isna(value):
                    # Skip if this value matches the paper title
                    if paper_title and paper_title.strip() == value.strip():
                        continue
                        
                    # Clean numbering patterns for ALL categories now
                    # Remove numbering patterns like "1)", "2) ", etc.
                    # First split by comma if there are multiple items
                    items = value.split(',')
                    cleaned_items = []
                    
                    for item in items:
                        # Remove numbering pattern (like "1) " or "1. " or "1 - ")
                        cleaned_item = re.sub(r'^\s*\d+[\)\.:\-\s]+\s*', '', item.strip())
                        if cleaned_item:
                            cleaned_items.append(cleaned_item)
                    
                    # Join all cleaned items back together with commas
                    if cleaned_items:
                        value = ', '.join(cleaned_items)
                    
                    # Skip if we've already added this value or a similar one
                    if value in used_values:
                        continue
                    
                    # Add to used values to prevent duplicates
                    used_values.add(value)
                    tags_by_type[priority].append((value, tag_type))
        
        # Then check if it's a Category (for fields like 'novel_findings', 'limitations')
        elif subcategory == '-' and category in df['Category'].values:
            rows = df[df['Category'] == category]
            if not rows.empty:
                value = rows[doc].iloc[0]
                if value and not pd.isna(value):
                    # Skip if this value matches the paper title
                    if paper_title and paper_title.strip() == value.strip():
                        continue
                        
                    # Just take the first ~30 characters for these as they can be lengthy
                    if len(value) > 30:
                        short_value = value[:30].strip() + "..."
                    else:
                        short_value = value
                    
                    # Skip if we've already added this value or a similar one
                    if short_value in used_values:
                        continue
                    
                    # Add to used values to prevent duplicates
                    used_values.add(short_value)
                    tags_by_type[priority].append((short_value, tag_type))
    
    # Combine the tags in the specified order
    ordered_tags = []
    ordered_tags.extend(tags_by_type['year'])
    ordered_tags.extend(tags_by_type['study_design'])
    ordered_tags.extend(tags_by_type['harmful'])
    ordered_tags.extend(tags_by_type['other'])
    
    # Limit to max 8 tags
    return ordered_tags[:8]
