# ExplanableAI

A Python-based project that implements explainable AI techniques by approximating clustering results using decision trees.

## Description

This project focuses on making clustering results more interpretable by using decision trees to approximate the clustering outcomes. This approach provides a more transparent and understandable way to explain how the clustering algorithm makes its decisions.

The project includes two main applications:

1. **ExplainableAI.py**: A basic implementation that approximates K-means clustering results with a decision tree to gain insights into the characteristics of each cluster.

2. **ExplainableAI2.py**: An advanced implementation that approximates clustering data after dimensionality reduction (using PCA and UMAP) with a decision tree that branches using the original features. This helps visualize and explain why data was divided in a particular way after dimensionality reduction, making the characteristics of each cluster more interpretable.

## Features

### ExplainableAI.py
- K-means clustering with customizable number of clusters
- Feature selection for clustering
- Decision tree approximation of clustering results
- Visualization of the decision tree structure


### ExplainableAI2.py
- K-means clustering with customizable number of clusters
- PCA dimensionality reduction with configurable components
- UMAP dimensionality reduction (2D or 3D visualization)
- Decision tree approximation using original features
- Visualization of clustering results in reduced dimensions
- Histogram visualization of feature distributions at each decision node
- Interactive Streamlit interface for data upload and parameter adjustment

## Requirements

- Python 3.8+
- Required Python packages:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - graphviz
  - umap-learn
  - streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ExplanableAI.git
cd ExplanableAI
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running ExplainableAI.py
```bash
streamlit run ExplanableAI/ExplainableAIApp.py
```

### Running ExplainableAI2.py
```bash
streamlit run ExplanableAI/ExplainableAIApp2.py
```

### Example Datasets

The repository includes two example datasets that you can use to test the applications:

1. **wine_data.csv**: A dataset containing various chemical properties of wines. This dataset is useful for testing clustering on numerical data with multiple features.

2. **Mall_Customers.csv**: A dataset containing customer information including age, annual income, and spending score. This dataset is useful for testing clustering on customer segmentation data.

To use these datasets, simply upload them when prompted by the application.

## How It Works

1. **Data Preparation**: Upload a CSV file with your data.

2. **Clustering**:
   - ExplainableAI.py: Select features and perform K-means clustering directly on the data.
   - ExplainableAI2.py: Perform PCA and UMAP dimensionality reduction before K-means clustering.

3. **Decision Tree Approximation**: 
   - The clustering results are approximated using a decision tree that branches based on the original features.
   - This helps understand which features are most important for distinguishing between clusters.

4. **Visualization**:
   - The decision tree structure is visualized using Graphviz.
   - ExplainableAI2.py also provides histograms of feature distributions at each decision node.
