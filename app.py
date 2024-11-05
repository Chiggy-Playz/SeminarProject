from matplotlib.patches import Rectangle
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns

try:
    data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("Please ensure 'dataset.csv' is in the same directory as this script.")
    exit()

RANDOM_STATE = 42
COLORS = ["#89CFF0", "#FF69B4", "#FFD700", "#7B68EE", "#FF4500",
          "#9370DB", "#32CD32", "#8A2BE2", "#FF6347", "#20B2AA",
          "#FF69B4", "#00CED1", "#FF7F50", "#7FFF00", "#DA70D6"]
class MultiModelRecommender:
    def __init__(self):
        self.label_encoders = {}
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVC': SVC(probability=True, random_state=RANDOM_STATE),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        }
        
        self.feature_columns = [
            'Age',
            'Gender',
        ]
        
        self.target_columns = [
            'Item Purchased',
            'Category',
            'Size',
            'Color'
        ]
    
    def preprocess_data(self, df):
        """Preprocess the data by encoding categorical variables"""
        processed_df = df.copy()
        
        processed_df['Size'] = processed_df['Size'].map({
            'S': 'Small',
            'M': 'Medium',
            'L': 'Large'
        })

        for column in self.feature_columns + self.target_columns:
            if processed_df[column].dtype == 'object':
                self.label_encoders[column] = LabelEncoder()
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
        
        return processed_df
    
    def fit(self, df, selected_model):
        """Train the selected model on the provided data"""
        processed_df = self.preprocess_data(df)
        
        X = processed_df[self.feature_columns]
        
        
        self.target_models = {}
        for target in self.target_columns:
            
            if selected_model == 'Random Forest':
                self.target_models[target] = RandomForestClassifier(n_estimators=100, random_state=42)
            elif selected_model == 'KNN':
                self.target_models[target] = KNeighborsClassifier(n_neighbors=5)
            elif selected_model == 'SVC':
                self.target_models[target] = SVC(probability=True, random_state=42)
            else:  
                self.target_models[target] = LogisticRegression(random_state=42, max_iter=1000)
                
            self.target_models[target].fit(X, processed_df[target])
    
    def predict(self, customer_data):
        """Predict preferences for a new customer"""
        processed_customer = pd.DataFrame([customer_data])
        
        for column in self.feature_columns:
            if column in self.label_encoders:
                processed_customer[column] = self.label_encoders[column].transform(processed_customer[column])
        
        predictions = {}
        
        for target in self.target_columns:
            model = self.target_models[target]
            pred_encoded = model.predict(processed_customer[self.feature_columns])
            pred_probs = model.predict_proba(processed_customer[self.feature_columns])[0]
            
            
            classes = self.label_encoders[target].classes_
            pred_with_probs = [(classes[i], prob) for i, prob in enumerate(pred_probs)]
            pred_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            predictions[target] = pred_with_probs[:5]
        
        return predictions

class CustomerSegmentation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=2)
        
        
        self.features = [
            'Age', 
            'Purchase Amount (USD)',
            'Review Rating',
            'Previous Purchases',
            'Gender',
            'Category',
            'Location',
            'Season',
            'Subscription Status',
            'Frequency of Purchases'
        ]
    
    def preprocess_data(self, df):
        """Preprocess the data for clustering"""
        processed_df = df.copy()
        for column in self.features:
            if processed_df[column].dtype == 'object':
                self.label_encoders[column] = LabelEncoder()
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
        
        scaled_features = self.scaler.fit_transform(processed_df[self.features])
        return scaled_features, processed_df
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        return inertias
    
    def fit(self, df, n_clusters):
        """Perform clustering"""
        scaled_features, processed_df = self.preprocess_data(df)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(scaled_features)
        df['Cluster'] = clusters
        pca_features = self.pca.fit_transform(scaled_features)
        return df, pca_features
    
    def get_cluster_insights(self, df):
        """Generate insights for each cluster"""
        insights = {}
        
        for cluster in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster]
            insights[cluster] = {
                'size': len(cluster_data),
                'avg_age': cluster_data['Age'].mean(),
                'avg_purchase': cluster_data['Purchase Amount (USD)'].mean(),
                'avg_rating': cluster_data['Review Rating'].mean(),
                'top_categories': cluster_data['Category'].value_counts().head(3),
                'top_colors': cluster_data['Color'].value_counts().head(3),
                'top_sizes': cluster_data['Size'].value_counts().head(3),
                'preferred_season': cluster_data['Season'].mode().iloc[0],
                'avg_previous_purchases': cluster_data['Previous Purchases'].mean(),
                'preferred_payment': cluster_data['Payment Method'].mode().iloc[0],
                'purchase_frequency': cluster_data['Frequency of Purchases'].mode().iloc[0]
            }
        
        return insights

def add_segmentation_section():
    st.header("📊 Customer Segmentation Analysis")
        
    
    segmentation = CustomerSegmentation()   
    n_clusters = st.sidebar.slider("Number of Customer Segments", 2, 8, 4)
    df_with_clusters, pca_features = segmentation.fit(data, n_clusters)
    insights = segmentation.get_cluster_insights(df_with_clusters)
    tab1, tab2 = st.tabs(["Segment Profiles", "Detailed Analysis"])
    
    with tab1:
        for cluster in sorted(insights.keys()):
            with st.expander(f"Segment {cluster + 1} Profile"):
                col1, col2 = st.columns(2)      
                with col1:
                    st.write("📊 **Basic Statistics**")
                    st.write(f"Segment Size: {insights[cluster]['size']} customers")
                    st.write(f"Average Age: {insights[cluster]['avg_age']:.1f} years")
                    st.write(f"Average Purchase: ${insights[cluster]['avg_purchase']:.2f}")
                    st.write(f"Average Rating: {insights[cluster]['avg_rating']:.1f}/5.0")
                
                with col2:
                    st.write("🛍️ **Shopping Preferences**")
                    st.write(f"Preferred Season: {insights[cluster]['preferred_season']}")
                    st.write(f"Purchase Frequency: {insights[cluster]['purchase_frequency']}")
                    st.write(f"Payment Method: {insights[cluster]['preferred_payment']}")
    
    with tab2:        
        col1, col2 = st.columns(2)        
        with col1:            
            fig_age = px.box(df_with_clusters, x='Cluster', y='Age', 
                           title="Age Distribution by Segment")
            st.plotly_chart(fig_age)
        
        with col2:            
            fig_purchase = px.box(df_with_clusters, x='Cluster', 
                                y='Purchase Amount (USD)',
                                title="Purchase Amount Distribution by Segment")
            st.plotly_chart(fig_purchase)
        
        col3, col4 = st.columns(2)
        
        with col3:            
            fig_previous = px.box(df_with_clusters, x='Cluster', 
                                y='Previous Purchases',
                                title="Previous Purchases by Segment")
            st.plotly_chart(fig_previous)
        
        with col4:
            
            fig_ratings = px.box(df_with_clusters, x='Cluster', 
                               y='Review Rating',
                               title="Review Ratings by Segment")
            st.plotly_chart(fig_ratings)

def add_prediction_section():
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a prediction model",
        ['Random Forest', 'KNN', 'SVC', 'Logistic Regression']
    )
    
    
    model_descriptions = {
        'Random Forest': 'Ensemble learning method that combines multiple decision trees. Good for handling both numerical and categorical data.',
        'KNN': 'Makes predictions based on the k-nearest neighbors in the training data. Simple but effective for pattern recognition.',
        'SVC': 'Support Vector Classification with probability estimates. Effective for finding optimal boundaries between classes.',
        'Logistic Regression': 'Simple but interpretable model that works well for binary and multiclass classification.'
    }
    st.sidebar.info(model_descriptions[selected_model])
    
    
    recommender = MultiModelRecommender()
    recommender.fit(data, selected_model)
    
    
    st.sidebar.header("Customer Information")
    
    age = st.sidebar.slider("Age", data["Age"].min(), data["Age"].max(), 30)
    gender = st.sidebar.selectbox("Gender",data["Gender"].unique())
    
    
    if st.sidebar.button("Predict Preferences"):
        with st.spinner(f'Making predictions using {selected_model}...'):
            customer_data = {
                'Age': age,
                'Gender': gender,
            }
            
            predictions = recommender.predict(customer_data)
            
            
            st.header("Predicted Preferences")
            
            
            st.subheader("🎯 Recommended Items")
            for item, prob in predictions['Item Purchased']:
                st.write(f"• {item} ({prob:.1%} confidence)")
            
            st.write("---")
            
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Size")
                pred, prob = predictions['Size'][0]
                st.write(f"{pred} ({prob:.1%} confidence)")
            
            with col2:
                for target in ['Color']:
                    st.subheader(f"{target}")
                    for pred, prob in predictions[target]:
                        st.write(f"• {pred} ({prob:.1%} confidence)")


def add_exploratory_data_analysis_section():
    st.header("🔍 Exploratory Data Analysis")
    
    st.write("This section provides an overview of the dataset.")
    
    st.subheader("Dataset Overview")
    st.write(data.head())
    
    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Descriptive Statistics")

    st.write(data.describe())

    st.subheader("Data Visualization")
    st.write("Visualizing the dataset using various plots.")

    # SNS Pairplot
    st.subheader("Pairplot")
    fig = plt.figure(figsize=(16, 10))
    sns.pairplot(data)
    st.pyplot(fig)

    st.write("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Gender"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Male', 'Female'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Gender', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)
    
    with col2:
        st.subheader("Gender Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Gender"].value_counts()
        explode = (0, 0.1)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Gender', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        plt.legend(labels=counts.index, loc="best")
        st.pyplot(fig_pie)
    
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Category"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Clothing', 'Accessories', 'Footwear', 'Outerwear'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height()), ha='center', va='bottom', color='black')
        plt.xlabel('Category', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)
    
    with col2:
        st.subheader("Category Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Category"].value_counts()
        explode = (0, 0.0, 0.0, 0.1)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Category', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        plt.legend(labels=counts.index, loc="best")
        st.pyplot(fig_pie)

    st.write("---")

    st.subheader("Item Purchased Distribution")

    fig = plt.figure(figsize=(16, 7))
    data["Item Purchased"].value_counts().sort_values(ascending=True).plot(kind='barh', color=sns.color_palette('tab20'), edgecolor='black')
    plt.ylabel('Item Purchased', fontsize=16)
    plt.xlabel('\nNumber of Occurrences', fontsize=16)
    plt.title('Item Purchased\n', fontsize=16)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    st.subheader("Location Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Location"].value_counts()[:10].sort_values(ascending=False).plot(kind='bar', color=sns.color_palette('inferno'), edgecolor='black')
    plt.xlabel('Location', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Size Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Size"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Medium', 'Large', 'Small', 'Extra Large'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Size', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)

    with col2:
        st.subheader("Size Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Category"].value_counts()
        explode = (0, 0.0, 0.0, 0.1)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Size', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        st.pyplot(fig_pie)
    
    st.write("---")

    st.subheader("Color Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Color"].value_counts().sort_values(ascending=True).plot(kind='barh', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Color', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Season Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Season"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Spring', 'Fall', 'Winter', 'Summer'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Season', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)  
    
    with col2:
        st.subheader("Season Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Season"].value_counts()
        explode = (0, 0, 0, 0)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Season', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        st.pyplot(fig_pie)
    
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Subscription Status Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Subscription Status"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('No', 'Yes'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Subscription Status', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)
    
    with col2:
        st.subheader("Subscription Status Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Subscription Status"].value_counts()
        explode = (0, 0.1)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Subscription Status', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        st.pyplot(fig_pie)
    
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Payment Method Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Payment Method"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Credit Card', 'Venmo', 'Cash', 'Paypal', 'Debit Card', 'Bank Transfer'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Payment Method', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)

    with col2:
        st.subheader("Payment Method Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Payment Method"].value_counts()
        explode = (0, 0, 0, 0, 0.0, 0.06)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Payment Method', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        st.pyplot(fig_pie)
    
    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shipping Type Distribution (Bar Chart)")
        fig_bar = plt.figure(figsize=(10, 6))
        ax = data["Shipping Type"].value_counts().plot(kind='bar', color=COLORS, rot=0)
        ax.set_xticklabels(('Free Shipping', 'Standard', 'Store Pickup', 'Next Day Air', 'Express', '2-Day Shipping'))
        for p in ax.patches:
            assert isinstance(p, Rectangle)
            ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha='center', va='bottom', color='black')
        plt.xlabel('Shipping Type', weight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
        st.pyplot(fig_bar)
    
    with col2:
        st.subheader("Shipping Type Distribution (Pie Chart)")
        fig_pie = plt.figure(figsize=(20, 6))
        counts = data["Shipping Type"].value_counts()
        explode = (0, 0, 0, 0, 0.0, 0.06)
        counts.plot(kind='pie', fontsize=12, colors=COLORS, explode=explode, autopct='%1.1f%%')
        plt.xlabel('Shipping Type', weight="bold", color="#2F0F5D", fontsize=14, labelpad=20)
        plt.axis('equal')
        st.pyplot(fig_pie)
    
    st.write("---")

    st.subheader("Frequency of Purchases Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Frequency of Purchases"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Frequency of Purchases', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    st.subheader("Review Rating Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Review Rating"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Review Rating', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    st.subheader("Promo Code Used Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Promo Code Used"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Promo Code Used', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    st.subheader("Discount Applied Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Discount Applied"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Discount Applied', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    st.pyplot(fig)

    st.write("---")

    st.subheader("Previous Purchases Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Previous Purchases"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Previous Purchases', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    st.pyplot(fig)

    st.write("---")

    st.subheader("Age Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Age"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Age', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("---")

    st.subheader("Purchase Amount Distribution")

    fig = plt.figure(figsize=(16, 6))
    data["Purchase Amount (USD)"].value_counts().sort_values(ascending=True).plot(kind='bar', color=sns.color_palette('tab20'), edgecolor='black')
    plt.xlabel('Purchase Amount (USD)', weight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Number of Occurrences', weight="bold", fontsize=14, labelpad=20)
    plt.xticks(rotation=0, ha='center', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.title("Customer Analytics Dashboard")
    tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Customer Predictions", "Exploratory Data Analysis"])
    
    with tab1:
        add_segmentation_section()
    
    with tab2:
        add_prediction_section()

    with tab3:
        add_exploratory_data_analysis_section()

if __name__ == "__main__":
    st.set_page_config(page_title="Customer Analytics")
    main()
