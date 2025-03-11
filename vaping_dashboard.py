import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Page config
st.set_page_config(
    page_title="IB GenAI R&D Tool",
    page_icon="🔬",
    layout="wide"
)

logo = Image.open("Images/IB-logo.png")
st.image(logo, width=200)

# Title
st.title("GenAI R&D Tool: E-Cigarette Research & Insights")

# Data from research papers
adverse_events_data = pd.DataFrame({
    'name': ['Sore/dry mouth', 'Cough', 'Mouth/tongue sores', 'Gingivitis', 'Headache', 'Dizziness', 'Heart palpitation'],
    'ecigaretteOnly': [3.5, 2.1, 2.2, 3.1, 1.5, 0.6, 0.9],
    'dualUser': [4.9, 13.1, 6.0, 3.8, 2.7, 2.7, 3.3]
})

health_improvement_data = pd.DataFrame({
    'name': ['Breathing', 'Smell', 'Taste', 'Physical status', 'Stamina', 'Mood', 'Sleep quality'],
    'ecigaretteOnly': [92.7, 90.0, 87.2, 84.1, 82.0, 54.5, 48.1],
    'dualUser': [77.8, 67.6, 66.7, 65.6, 57.5, 35.2, 34.8]
})

oral_health_impact_data = pd.DataFrame({
    'name': ['Periodontitis', 'Caries risk', 'Gingivitis', 'Oral mucosal changes', 'Other oral effects'],
    'value': [35, 25, 20, 15, 5]
})

research_focus_data = pd.DataFrame({
    'year': ['2017', '2018', '2019', '2020', '2021', '2022'],
    'periodontal': [15, 20, 25, 30, 35, 40],
    'caries': [5, 8, 10, 15, 18, 25],
    'sensation': [10, 15, 20, 22, 25, 30],
    'respiratory': [20, 25, 30, 35, 38, 40]
})

user_perception_data = pd.DataFrame({
    'name': ['E-cigarette only', 'Dual user'],
    'healthImproved': [82, 65],
    'adverseEvents': [12, 26],
    'noChange': [6, 9]
})

# New data for additional tabs
publications_health_outcomes = pd.DataFrame({
    'outcome': ['Popcorn Lungs', 'Myocardial infarction'],
    'count': [8, 5]
})

publications_causal_factors = pd.DataFrame({
    'factor': ['Diacetyl / Hydroxide', 'Heating Temperature > 200C'],
    'count': [7, 4]
})

publications_data = pd.DataFrame({
    'name': ['Javed et al. (2017)', 'Ghazali et al. (2018)', 'Ye et al. (2020)', 'Jeong et al. (2020)', 
             'Velmulapalli et al. (2021)', 'Irusa et al. (2022)', 'Ramenzoni et al. (2022)', 'Xu et al. (2022)'],
    'author_journal': ['Javed et al. / Journal of Periodontology', 'Ghazali et al. / Journal of International Dental Medical Research',
                       'Ye et al. / Journal of Periodontology', 'Jeong et al. / Journal of Periodontology',
                       'Velmulapalli et al. / Journal of American Dental Association', 'Irusa et al. / Journal of American Dental Association',
                       'Ramenzoni et al. / Toxics', 'Xu et al. / Dental Journal'],
    'date': ['2017', '2018', '2020', '2020', '2021', '2022', '2022', '2022'],
    'key_insights': ['Periodontal inflammation and self-perceived oral symptoms were poorer among cigarette smokers than vapers',
                     'E-cigarettes have potentially detrimental effects on oral health',
                     'Smoking/vaping produces significant effects on oral health',
                     'Smoking and vaping produce incremented rates of periodontal disease',
                     'Vaping and dual smoking associated with increased occurrence of untreated caries',
                     'Vaping patients had a higher risk of developing caries',
                     'E-cig smoking may contribute to cell damage of oral tissue and inflammation',
                     'Flavored e-liquids have a more detrimental impact on oral commensal bacteria']
})

contradictions_data = pd.DataFrame({
    'study': ['Study A (2018)', 'Study B (2019)', 'Study C (2020)', 'Study D (2021)', 'Study E (2022)'],
    'sample_size': [120, 350, 1042, 4618, 13216],
    'methodology': ['Cross-sectional', 'Case-control', 'Cohort', 'Cross-sectional', 'In vitro'],
    'finding': ['Minimal oral health impact', 'Significant periodontal risks', 'Mixed effects based on user type', 
                'Increased caries risk', 'Cellular damage to oral tissues'],
    'significance': ['p > 0.05', 'p < 0.01', 'p < 0.05', 'p < 0.001', 'N/A'],
    'conflict_index': [0.2, 0.7, 0.5, 0.8, 0.6]
})

bias_data = pd.DataFrame({
    'study': ['Industry Study 1', 'Industry Study 2', 'Academic Study 1', 'Academic Study 2', 'Government Study'],
    'funding_source': ['Tobacco Industry', 'E-cigarette Manufacturer', 'University Grant', 'National Health Grant', 'FDA'],
    'biases_detected': [3, 4, 1, 0, 0],
    'positive_findings': [92, 85, 45, 30, 28],
    'negative_findings': [8, 15, 55, 70, 72],
    'sample_quality': [2, 3, 4, 5, 5]
})

# Tabs
tabs = st.tabs(["Overview", "Adverse Events", "Perceived Benefits", "Oral Health", "Research Trends", 
                "Contradictions & Conflicts", "Bias in Research", "Publication Level"])

# Overview Tab
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Key Findings")
        st.markdown("""
        - E-cigarette-only users report fewer adverse events (11.8%) compared to dual users (26.2%)
        - Sensory improvements (smell, taste) are reported by 90% of e-cigarette-only users
        - Vaping is associated with increased risk of periodontitis and caries
        - Flavored e-liquids have more detrimental effects on oral bacteria than unflavored ones
        - 82.4% of surveyed users were e-cigarette-only users, while 17.6% were dual users
        """)
    
    with col2:
        st.subheader("User Perception (% reporting)")
        
        # Convert to long format for Plotly
        user_perception_long = pd.melt(
            user_perception_data, 
            id_vars=['name'], 
            value_vars=['healthImproved', 'adverseEvents', 'noChange'],
            var_name='category', 
            value_name='percentage'
        )
        
        # Map category names to more readable versions
        category_map = {
            'healthImproved': 'Health Improved',
            'adverseEvents': 'Adverse Events',
            'noChange': 'No Change'
        }
        user_perception_long['category'] = user_perception_long['category'].map(category_map)
        
        fig = px.bar(
            user_perception_long,
            x='percentage',
            y='name',
            color='category',
            orientation='h',
            title="User Perception",
            labels={'percentage': 'Percentage (%)', 'name': '', 'category': 'Response'},
            color_discrete_sequence=['#82ca9d', '#ff7675', '#74b9ff']
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(range=[0, 100]),
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Demographics from Studies")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        st.markdown("#### Gender Distribution")
        st.markdown("From analyzed studies: 82.9% male, 17.1% female")
    
    with demo_col2:
        st.markdown("#### Age Range")
        st.markdown("Mean age across studies: 35.0 ± 7.6 years")
    
    with demo_col3:
        st.markdown("#### Previous Smoking Habits")
        st.markdown("""
        - 60.6% heavy smokers (≥20 cigarettes/day)
        - 29.6% moderate smokers (11-19 cigarettes/day)
        - 9.7% light smokers (≤10 cigarettes/day)
        """)


# Adverse Events Tab
with tabs[1]:
    st.subheader("Adverse Events: E-cigarette-only vs. Dual Users (%)")
    st.markdown("Data shows dual users report higher rates of most adverse events compared to e-cigarette-only users")
    
    # Convert data to long format for Plotly
    adverse_events_long = pd.melt(
        adverse_events_data, 
        id_vars=['name'], 
        value_vars=['ecigaretteOnly', 'dualUser'],
        var_name='user_type', 
        value_name='percentage'
    )
    
    # Map user types to more readable versions
    user_type_map = {
        'ecigaretteOnly': 'E-cigarette Only',
        'dualUser': 'Dual User'
    }
    adverse_events_long['user_type'] = adverse_events_long['user_type'].map(user_type_map)
    
    fig = px.bar(
        adverse_events_long,
        x='name',
        y='percentage',
        color='user_type',
        barmode='group',
        title="Adverse Events by User Type",
        labels={'percentage': 'Percentage (%)', 'name': 'Adverse Event', 'user_type': 'User Type'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Adverse Event Findings")
    st.markdown("""
    - Cough is the most common adverse event in dual users (13.1%)
    - Sore/dry mouth and throat is most common in e-cigarette-only users (3.5%)
    - Significantly more dual users experience cough, mouth sores, dizziness and heart palpitation
    - Overall adverse events: 26.2% of dual users vs 11.8% of e-cigarette-only users
    - Mouth/throat-related adverse events are most common among e-cigarette-only users
    """)


# Perceived Benefits Tab
with tabs[2]:
    st.subheader("Perceived Health Improvements (%)")
    st.markdown("E-cigarette-only users consistently report greater health improvements than dual users")
    
    # Convert data to long format for Plotly
    health_improvement_long = pd.melt(
        health_improvement_data, 
        id_vars=['name'], 
        value_vars=['ecigaretteOnly', 'dualUser'],
        var_name='user_type', 
        value_name='percentage'
    )
    
    # Map user types to more readable versions
    health_improvement_long['user_type'] = health_improvement_long['user_type'].map(user_type_map)
    
    fig = px.bar(
        health_improvement_long,
        x='name',
        y='percentage',
        color='user_type',
        barmode='group',
        title="Perceived Health Improvements by User Type",
        labels={'percentage': 'Percentage (%)', 'name': 'Health Aspect', 'user_type': 'User Type'},
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Improvement Factors")
        st.markdown("""
        Research identified three main dimensions of perceived health improvements:
        
        1. **Sensory improvement:** Better smell and taste (highest improvement rates)
        2. **Physical functioning:** Improved breathing, physical well-being, and stamina
        3. **Mental health improvement:** Better mood, sleep quality, appetite, and memory
        """)
    
    with col2:
        st.subheader("Predictors of Greater Health Improvements")
        st.markdown("""
        - Being an e-cigarette-only user (vs. dual user)
        - Longer duration of e-cigarette use (>1 year)
        - Higher intensity of past smoking (≥20 cigarettes per day)
        - Male gender (for sensory improvements)
        """)


# Oral Health Tab
with tabs[3]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Oral Health Impact Categories")
        
        fig = px.pie(
            oral_health_impact_data,
            values='value',
            names='name',
            title="Oral Health Impact Categories",
            color_discrete_sequence=['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8']
        )
        
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Oral Health Findings")
        st.markdown("""
        - Vaping is associated with increased risk of periodontal disease
        - Higher risk of developing caries in vaping patients compared to non-users
        - Flavored e-liquids have more detrimental effects on oral bacteria than unflavored ones
        - E-cigarette smoking may contribute to oral tissue cell damage and inflammation
        - Some studies found hyperplastic candidiasis as a common oral mucosal lesion among e-cigarette users
        """)
    
    st.subheader("Comparison Between User Types")
    
    # Create DataFrame for comparison table
    comparison_data = {
        'Factor': ['Periodontal inflammation', 'Self-perceived oral symptoms', 'Caries risk', 'Inflammatory biomarkers in saliva'],
        'E-cigarette Only': ['Moderate', 'Moderate', 'Increased', 'Elevated'],
        'Dual Users': ['High', 'High', 'High', 'Highly elevated'],
        'Non-users': ['Low', 'Low', 'Baseline', 'Normal']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Custom styling for the table
    def highlight_cells(val):
        if val == 'High' or val == 'Highly elevated':
            return 'background-color: #FFCCCB'
        elif val == 'Moderate' or val == 'Increased' or val == 'Elevated':
            return 'background-color: #FFFFCC'
        elif val == 'Low' or val == 'Baseline' or val == 'Normal':
            return 'background-color: #CCFFCC'
        else:
            return ''
    
    st.dataframe(
        comparison_df.style.applymap(highlight_cells, subset=pd.IndexSlice[:, ['E-cigarette Only', 'Dual Users', 'Non-users']]),
        use_container_width=True
    )


# Research Trends Tab
with tabs[4]:
    st.subheader("Research Focus Areas (2017-2022)")
    st.markdown("Showing the evolution of research emphasis in vaping studies")
    
    fig = px.line(
        research_focus_data,
        x='year',
        y=['periodontal', 'caries', 'sensation', 'respiratory'],
        markers=True,
        labels={
            'value': 'Research Activity (relative units)',
            'variable': 'Research Area',
            'year': 'Year'
        },
        title="Evolution of Research Focus Areas",
        color_discrete_map={
            'periodontal': '#3498db',
            'caries': '#e74c3c',
            'sensation': '#2ecc71',
            'respiratory': '#f39c12'
        }
    )
    
    # Rename the lines
    fig.for_each_trace(lambda t: t.update(name = {
        'periodontal': 'Periodontal Health',
        'caries': 'Caries Research',
        'sensation': 'Sensory Studies',
        'respiratory': 'Respiratory Effects'
    }.get(t.name, t.name)))
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emerging Research Priorities")
        st.markdown("""
        - Longitudinal studies on long-term health effects
        - Impact of different e-liquid flavors on oral microbiome
        - Molecular mechanisms of vapor-induced inflammation
        - Comparative studies between different vaping devices
        - Effects of vaping on wound healing and tissue regeneration
        """)
    
    with col2:
        st.subheader("Limitations in Current Research")
        st.markdown("""
        - Self-reported data prone to recall and social desirability bias
        - Respondent bias (satisfied users more likely to participate)
        - Cross-sectional designs limiting causal inference
        - Inability to separate impact of positive expectancies from actual effects
        - Limited representative sampling of e-cigarette users
        - Lack of standardization in product types and usage patterns
        """)


# Contradictions & Conflicts Tab
with tabs[5]:
    st.subheader("Contradictions and Conflicts in Vaping Research")
    st.markdown("""
    Research on vaping health effects shows significant contradictions across studies.
    This analysis examines methodological differences that may explain conflicting results.
    """)
    
    # Contradiction comparison
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Conflicting Study Comparisons")
        
        # Style the table to highlight contradictions
        def highlight_contradictions(s):
            if s.name == 'finding':
                return ['background-color: #ffcccb' if 'Minimal' in x else 
                        'background-color: #ffffcc' if 'Mixed' in x else 
                        'background-color: #ffcccb' if 'Increased' in x or 'Significant' in x or 'damage' in x else '' 
                        for x in s]
            elif s.name == 'conflict_index':
                return ['background-color: #ccffcc' if x < 0.3 else 
                        'background-color: #ffffcc' if 0.3 <= x < 0.6 else 
                        'background-color: #ffcccb' if x >= 0.6 else '' 
                        for x in s]
            else:
                return ['' for _ in s]
        
        # Display styled table
        st.dataframe(
            contradictions_data.style.apply(highlight_contradictions),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Methodology for Capturing Conflicts")
        st.markdown("""
        **Data Sources:**
        - Research papers, reviews, meta-analyses
        - Examination of methodological differences
        
        **Comparison Points:**
        - Methodology & study design
        - Sample size & composition
        - Statistical approaches
        - Data collection techniques
        
        **Conflict Index:**
        The conflict index measures contradiction level with other studies (0-1 scale):
        - <0.3: Minimal conflict
        - 0.3-0.6: Moderate conflict
        - >0.6: Significant conflict
        """)
    
    # Contradiction visualization
    st.subheader("Visualization of Contradicting Study Findings")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(
            x=contradictions_data['study'],
            y=contradictions_data['sample_size'],
            name="Sample Size",
            marker_color='#3498db'
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=contradictions_data['study'],
            y=contradictions_data['conflict_index'],
            name="Conflict Index",
            mode="lines+markers",
            marker=dict(size=10),
            line=dict(width=4, color='#e74c3c')
        ),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text="Study Sample Size vs. Conflict Index",
        height=500
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Study")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Sample Size", secondary_y=False)
    fig.update_yaxes(title_text="Conflict Index (0-1)", range=[0, 1], secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key patterns section
    st.subheader("Key Patterns in Research Contradictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Common Sources of Contradictions:**
        
        1. **Sampling Issues**
           - Different user populations (e.g., heavy vs. light former smokers)
           - Dual users vs. e-cigarette-only users not clearly separated
           - Selection bias in recruiting participants
        
        2. **Methodological Variations**
           - Different e-cigarette devices and liquids tested
           - Varying exposure assessment approaches
           - Inconsistent outcome measurements
        """)
    
    with col2:
        st.markdown("""
        **Statistical Considerations:**
        
        1. **Significance Thresholds**
           - Varying p-value cutoffs (0.05, 0.01, 0.001)
           - Some studies report trends rather than significant results
        
        2. **Control Factors**
           - Different baseline comparisons (non-smokers vs. former smokers)
           - Varying control for confounding variables
           - Inconsistent adjustment for prior smoking history
        """)


# Bias in Research Tab
with tabs[6]:
    st.subheader("Bias Analysis in Vaping Research")
    st.markdown("""
    Funding sources and researcher affiliations can influence study outcomes.
    This analysis examines potential biases in the vaping research literature.
    """)
    
    # Bias indicators graph
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Funding Source and Outcome Correlation")
        
        # Create a stacked bar chart for positive vs. negative findings
        bias_data_melted = pd.melt(
            bias_data,
            id_vars=['study', 'funding_source'],
            value_vars=['positive_findings', 'negative_findings'],
            var_name='finding_type',
            value_name='percentage'
        )
        
        # Rename categories to be more readable
        bias_data_melted['finding_type'] = bias_data_melted['finding_type'].map({
            'positive_findings': 'Positive Health Effects',
            'negative_findings': 'Negative Health Effects'
        })
        
        fig = px.bar(
            bias_data_melted,
            x='study',
            y='percentage',
            color='finding_type',
            barmode='stack',
            labels={'percentage': 'Percentage of Findings (%)', 'study': 'Study', 'finding_type': 'Finding Type'},
            color_discrete_sequence=['#82ca9d', '#ff7675'],
            hover_data=['funding_source']
        )
        
        fig.update_layout(
            title="Distribution of Positive vs. Negative Findings by Study Type",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bias Indicators")
        st.markdown("""
        **Key indicators of potential bias:**
        
        - **P-Hacking:** Multiple statistical tests without correction.
        
        - **Representative Sampling Issues:** Small, biased samples that don't reflect true user populations.
        
        - **Conflict of Interest:** Author affiliations with tobacco or e-cigarette industry.
        
        - **Selective Reporting:** Only reporting favorable outcomes while omitting negative findings.
        
        - **Author Reputation:** Past retractions, citation patterns suggesting bias.
        """)
    
    # Bias metrics visualization
    st.subheader("Metrics of Research Quality and Bias")
    
    # Create a scatter plot of sample quality vs. biases detected
    fig = px.scatter(
        bias_data,
        x='sample_quality',
        y='biases_detected',
        size='biases_detected',
        color='funding_source',
        hover_name='study',
        text='study',
        size_max=20,
        labels={
            'sample_quality': 'Sample Quality (1-5 scale)',
            'biases_detected': 'Number of Biases Detected',
            'funding_source': 'Funding Source'
        },
        title="Research Quality Assessment"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bias explanations
    st.subheader("Common Bias Patterns in Vaping Research")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Industry-Funded Research Patterns:**
        
        - Focus on short-term rather than long-term effects
        - Comparison primarily to combustible cigarettes, not to non-use
        - Emphasis on harm reduction rather than absolute harm
        - Selection of participants more likely to report positive outcomes
        - Use of subjective rather than objective outcome measures
        """)
    
    with col2:
        st.markdown("""
        **Academic/Government Research Patterns:**
        
        - More likely to include non-users as primary comparison group
        - Greater focus on potential negative health outcomes
        - More emphasis on youth uptake and addiction concerns
        - More longitudinal designs examining longer-term effects
        - Higher likelihood of reporting null findings
        """)
    
    # Recommendations for interpreting research
    st.subheader("Recommendations for Interpreting Vaping Research")
    st.markdown("""
    When evaluating vaping research, consider:
    
    1. **Funding source** and potential conflicts of interest
    2. **Sample size and composition** (representativeness)
    3. **Comparison groups** used (smokers, non-smokers, former smokers)
    4. **Length of study** (short-term vs. longitudinal effects)
    5. **Objective vs. subjective** outcome measures
    6. **Statistical approaches** and potential for p-hacking
    7. **Publication bias** (negative findings less likely to be published)
    """)


# Publication Level Tab
with tabs[7]:
    st.subheader("Publication Level Analysis")
    st.markdown("""
    This analysis examines publication patterns in vaping health effects research,
    including mentions of specific health outcomes and causal factors.
    """)
    
    # Health outcomes and causal factors graphs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mention of Health Outcomes")
        
        # Create a bar chart for health outcomes mentions
        fig = px.bar(
            publications_health_outcomes,
            x='outcome',
            y='count',
            labels={'outcome': 'Health Outcome', 'count': 'Number of Publications'},
            color='count',
            color_continuous_scale='Blues',
            title="Health Outcomes Mentioned in Publications"
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mention of Causal Factors")
        
        # Create a bar chart for causal factors mentions
        fig = px.bar(
            publications_causal_factors,
            x='factor',
            y='count',
            labels={'factor': 'Causal Factor', 'count': 'Number of Publications'},
            color='count',
            color_continuous_scale='Blues',
            title="Causal Factors Mentioned in Publications"
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Publications table
    st.subheader("Key Publications and Insights")
    
    # Create table of publications
    st.dataframe(
        publications_data,
        column_config={
            "name": st.column_config.TextColumn("Publication Name"),
            "author_journal": st.column_config.TextColumn("Publication Author / Journal"),
            "date": st.column_config.TextColumn("Date of Publish"),
            "key_insights": st.column_config.TextColumn("Key Harm Insights"),
        },
        use_container_width=True
    )
    
    # Publication trends
    st.subheader("Publication Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Temporal Trends:**
        
        - **2017-2018:** Initial focus on comparing vaping to smoking
        - **2019-2020:** Growing emphasis on specific oral health impacts
        - **2021-2022:** More targeted studies on specific issues (caries, tissue damage, cellular effects)
        - **Emerging (2023+):** Focus on long-term health outcomes and molecular mechanisms
        """)
    
    with col2:
        st.markdown("""
        **Methodological Evolution:**
        
        - Earlier studies relied more on self-reported data
        - Recent shift toward in vitro cellular studies
        - Increasing use of biomarkers in clinical research
        - Growing emphasis on standardized measurement tools
        - More longitudinal designs being implemented
        """)
    
    # Citation network and research gaps
    st.subheader("Research Gaps and Future Directions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Under-Researched Areas:**
        
        - Long-term effects of vaping (>5 years)
        - Interaction between vaping and existing oral conditions
        - Effects on dental materials and restorations
        - Impact on wound healing after dental procedures
        - Standardized clinical assessment protocols
        - Effects of newer generation devices
        """)
    
    with col2:
        st.markdown("""
        **Recommendations for Future Research:**
        
        - Conduct more longitudinal studies on health outcomes
        - Standardize protocols for measuring health effects
        - Establish dose-response relationships for various effects
        - Compare different e-liquid compositions and device types
        - Study effects in vulnerable populations (e.g., patients with existing oral disease)
        - Develop validated risk assessment tools for clinical use
        """)
        
        
# Add a sidebar with additional information
with st.sidebar:
    st.header("About This Dashboard")
    st.markdown("""
    This dashboard presents analysis of research on vaping health effects, 
    specifically focused on e-cigarette use patterns and health impacts.
    
    **Data sources:**
    - Research papers from peer-reviewed literature
    - Survey data from 31,647 participants
    - Mean participant age: 35.0 ± 7.6 years
    
    **Key topics covered:**
    - Adverse events
    - Perceived health benefits
    - Oral health impacts
    - Research trends
    - Contradictions & conflicts
    - Research bias analysis
    - Publication patterns
    
    **Last updated:** March, 2024
    """)
    
    st.markdown("---")
    
    # Add a filter that doesn't actually filter but demonstrates what could be done
    st.subheader("Filters")
    st.markdown("*(Demo only - not functional)*")
    
    # Date range filter
    st.date_input("Date Range", [pd.to_datetime("2017-01-01"), pd.to_datetime("2022-12-31")])
    
    # User type filter
    st.multiselect("User Type", ["E-cigarette Only", "Dual User"], default=["E-cigarette Only", "Dual User"])
    
    # Publication type filter
    st.multiselect("Publication Type", ["Peer-reviewed", "Clinical Trial", "Survey", "Review"], 
                  default=["Peer-reviewed", "Clinical Trial", "Survey"])
    
    # Funding source filter (new)
    st.multiselect("Funding Source", ["Industry", "Academic", "Government", "Non-profit"], 
                  default=["Industry", "Academic", "Government"])
    
    # Study design filter (new)
    st.multiselect("Study Design", ["Cross-sectional", "Cohort", "Case-control", "In vitro", "Clinical trial"], 
                  default=["Cross-sectional", "Cohort", "Case-control"])
    
    # Sample size range (new)
    st.slider("Sample Size Range", min_value=50, max_value=15000, value=(100, 10000))
    
    st.markdown("---")
    
    # Methodology notes
    st.subheader("Methodology Notes")
    st.markdown("""
    **Conflict Index Calculation:**
    The conflict index measures the degree to which a study's findings contradict other research in the field:
    
    - Literature review identifies major consensuses
    - Each study is scored based on deviation from consensus
    - Factors include sample quality, methodology, and findings
    - Higher score = greater contradiction with established findings
    
    **Bias Detection:**
    Research bias is identified through multiple indicators:
    
    - Funding source and disclosure analysis
    - Methodology assessment against standard protocols
    - Statistical analysis review for p-hacking
    - Selective reporting evaluation
    - Comparison of findings to funding source patterns
    """)