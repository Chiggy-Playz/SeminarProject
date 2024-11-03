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


try:
    data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("Please ensure 'dataset.csv' is in the same directory as this script.")
    exit()

RANDOM_STATE = 42

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
            
            predictions[target] = pred_with_probs[:5    ]
        
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
    tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Segment Profiles", "Detailed Analysis"])
    
    with tab1:    
        fig = px.scatter(
            x=pca_features[:, 0],
            y=pca_features[:, 1],
            color=df_with_clusters['Cluster'].astype(str),
            title="Customer Segments Visualization",
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        st.plotly_chart(fig)
        
        st.info("""
        This plot shows customer segments in 2D space. Closer points represent customers with similar characteristics.
        Each color represents a different customer segment.
        """)
    
    with tab2:
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
    
    with tab3:        
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
                        

def main():
    st.title("Customer Analytics Dashboard")
    tab1, tab2 = st.tabs(["Customer Predictions", "Customer Segmentation"])
    
    with tab1:
        add_segmentation_section()
    
    with tab2:
        add_prediction_section()

if __name__ == "__main__":
    st.set_page_config(page_title="Customer Analytics")
    main()