import pandas as pd
import streamlit as st

def display_document_details(df, matching_docs):
    """
    Display detailed information about the filtered documents.
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    matching_docs (list): List of document column names that match current filters
    """
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
        
        # Reset index and add a new index column starting from 1
        doc_details = doc_details.reset_index(drop=True)
        doc_details.index = doc_details.index + 1
        
        st.write(doc_details)
    else:
        st.write("No documents match the current filters")


def display_raw_data(df):
    """
    Display sample raw data from the DataFrame, focusing on the most complete documents.
    
    Parameters:
    df (pandas.DataFrame): Main DataFrame containing research data
    """
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
        sample_data = df[display_columns].copy()
        
        # Reset index and add a new index column starting from 1
        sample_data = sample_data.reset_index(drop=True)
        sample_data.index = sample_data.index + 1
        
        st.write(sample_data)
    else:
        st.write("No document data available")