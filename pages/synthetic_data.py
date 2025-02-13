import streamlit as st
import pandas as pd
import numpy as np

def generate_synthetic_data(df, num_rows):
    """
    Generate synthetic data based on the distribution of the uploaded dataset.
    
    For numeric columns, synthetic data is generated from a normal distribution 
    (using the column's mean and standard deviation). For non-numeric columns, 
    random sampling from the unique values is performed.
    """
    synthetic_df = pd.DataFrame()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            # If the std is zero or NaN, fill with the mean value
            if std == 0 or np.isnan(std):
                synthetic_df[col] = [mean] * num_rows
            else:
                synthetic_df[col] = np.random.normal(mean, std, num_rows)
        else:
            values = df[col].dropna().unique()
            if len(values) > 0:
                synthetic_df[col] = np.random.choice(values, num_rows)
            else:
                synthetic_df[col] = [np.nan] * num_rows
    return synthetic_df

def app():
    st.markdown(
        """
        <style>
        .main {background-color: #F5F5F5; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 24px;
            border-radius: 5px;
        }
        .stDownloadButton>button {
            background-color: #008CBA;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 24px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Synthetic Data Generation App")
    st.write("Upload your CSV file below to explore your data and generate synthetic data.")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
        
        # Data Overview Section
        st.subheader("Data Overview")
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")
        st.dataframe(data.head())
        
        # Data Properties Section
        st.subheader("Data Properties")
        st.write("**Missing Values per Column:**")
        st.write(data.isnull().sum())
        st.write("**Data Types:**")
        st.write(data.dtypes)
        
        st.markdown("---")
        st.subheader("Generate Synthetic Data")
        num_rows = st.number_input("Enter the number of synthetic rows to generate:", min_value=1, value=100, step=1)
        
        if st.button("Generate Synthetic Data"):
            synthetic_data = generate_synthetic_data(data, num_rows)
            st.success(f"Generated {num_rows} synthetic rows!")
            st.dataframe(synthetic_data.head())
            
            # Provide option to download the synthetic data as CSV
            csv = synthetic_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Synthetic Data as CSV",
                data=csv,
                file_name="synthetic_data.csv",
                mime="text/csv"
            )
    else:
        st.info("Awaiting CSV file upload.")
    