import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Tips Dashboard", layout="wide")

# Sidebar
st.sidebar.title("💡 Tips Data Visualizer")
uploaded_file = st.sidebar.file_uploader("📂 Upload your tips.csv file", type="csv")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app visualizes the Tips dataset with histograms, pie charts, "
    "boxplots, and more. Upload a valid CSV with columns like `total_bill`, `tip`, `sex`, and `day`."
)

st.title("🍽️ Restaurant Tips Data Dashboard")
st.markdown("Upload a CSV file to explore the tipping behavior of restaurant customers.")

# Main app
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Column validation
        required_cols = {'total_bill', 'tip', 'sex', 'day'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ CSV must include the columns: {', '.join(required_cols)}")
        else:
            # Show raw data preview
            if st.checkbox("📋 Show raw data"):
                st.dataframe(df.head())

            # Create 2x2 plots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("📊 Visual Analysis of Tips Dataset", fontsize=16, weight='bold')

            # 1. Histogram
            axs[0, 0].hist(df['total_bill'], bins=20, color='skyblue', edgecolor='black')
            axs[0, 0].set_title('Total Bill Distribution')
            axs[0, 0].set_xlabel('Total Bill')
            axs[0, 0].set_ylabel('Frequency')

            # 2. Boxplot
            axs[0, 1].boxplot(df['tip'], vert=True)
            axs[0, 1].set_title('Tip Amount Boxplot')
            axs[0, 1].set_ylabel('Tip')

            # 3. Pie Chart
            gender_counts = df['sex'].value_counts()
            axs[1, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            axs[1, 0].set_title('Gender Distribution')

            # 4. Bar chart by day
            avg_total_by_day = df.groupby('day')['total_bill'].mean()
            axs[1, 1].bar(avg_total_by_day.index, avg_total_by_day.values, color='lightgreen')
            axs[1, 1].set_title('Average Total Bill by Day')
            axs[1, 1].set_xlabel('Day')
            axs[1, 1].set_ylabel('Avg Total Bill')

            # Final layout and display
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Something went wrong while reading the file: {e}")
else:
    st.warning("👈 Please upload a CSV file from the sidebar to begin.")
