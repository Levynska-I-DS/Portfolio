![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*t98V5s6uNKVNEde5ZYQemw.jpeg)

# MovieLens Movie Recommendation System

### Project Overview

**Objective:** The goal of this project is to build a movie recommendation system using machine learning techniques. The system leverages the MovieLens dataset to predict user ratings for movies and provide personalized recommendations.

**Context:** Building a recommendation system involves handling large datasets and complex algorithms to predict user preferences accurately. Challenges include dealing with sparse data, ensuring scalability, and optimizing the recommendation algorithms for accuracy and performance.

**Significance:** Accurate movie recommendations can enhance user experience by providing personalized content, which is crucial for streaming services and content platforms. This project aims to improve the relevance of recommendations and can benefit further research in collaborative filtering and matrix factorization techniques.

**Goal:** The primary aim is to develop a robust recommendation system that can predict user ratings and suggest movies based on past interactions. The project also explores different models and parameters to enhance prediction accuracy and efficiency.

## Team Members

- **Iryna:** [GitHub](https://github.com/Levynska-I-DS)
- **Christian:** [GitHub](https://github.com/Kriss-fullstack)
- **Semih:** [GitHub](https://github.com/semihd97)

## Jupyter Notebooks

This project consists of several Jupyter Notebooks, each serving different purposes:

1. **Data_Preprocessing.ipynb:**  This notebook covers the initial data exploration and preprocessing steps. It includes loading the MovieLens dataset, handling missing values, performing exploratory data analysis (EDA), and preparing the data for model training.

2. **Model_Training.ipynb:** This notebook focuses on training various recommendation models, including collaborative filtering and matrix factorization. It includes hyperparameter tuning and model evaluation to select the best-performing model.

3. **Evaluation_and_Predictions.ipynb:** This notebook evaluates the performance of the trained models using metrics such as RMSE and MAE. It also includes code for generating movie recommendations for users based on the final model.

4. **interface_method_model_ML_GUI.ipynb:** This notebook provides a detailed description and usage of the graphical user interface (GUI) that interacts with the machine learning model. It explains the purpose of the GUI, the process of running it from a Python script, and how it integrates with the recommendation system. The notebook does not contain executable code for the GUI but provides information on how to create and use the GUI in a separate Python file.

5. **PowerBI_Analysis.ipynb:** This notebook includes Power BI integration for visualizing and analyzing the recommendation system's performance and insights. It helps in understanding data trends and evaluating model effectiveness through interactive dashboards.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MovieLens-Recommendation-System.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MovieLens-Recommendation-System
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the MovieLens dataset and place it in the `data` directory. The dataset can be acquired from [MovieLens](https://grouplens.org/datasets/movielens/).



**Note:** If any of the above files are missing, the corresponding functionality may not work as expected.

Once the setup is complete, you can use the provided Jupyter Notebooks to preprocess data, train models, and make predictions.

## Dataset

The dataset used in this project is the MovieLens dataset, which contains movie ratings provided by users. The data includes user ratings, movie metadata, and timestamps. You can download the dataset from [MovieLens](https://grouplens.org/datasets/movielens/).

## Attribute Information

The dataset contains the following attributes:

1. **userId:** Identifier for the user.
2. **movieId:** Identifier for the movie.
3. **rating:** User rating for the movie (ranging from 0.5 to 5.0).
4. **timestamp:** Time when the rating was given.

The dataset contains a total of 33,832,162 ratings and 86,537 movies. The ratings dataset includes all individual ratings, while the movies dataset provides metadata for each movie.

## EDA/Cleaning

The initial exploratory data analysis (EDA) includes visualizing the distribution of ratings, understanding the sparsity of the user-movie matrix, and identifying any data quality issues. Data cleaning steps involve handling missing values, normalizing ratings, and splitting the data into training and testing sets.

## Model Choices

We evaluated three recommendation models:
- **SVD (Singular Value Decomposition):** A matrix factorization technique that approximates user-item interactions by decomposing the rating matrix.
- **NMF (Non-negative Matrix Factorization):** Another matrix factorization method focused on non-negative values, offering a different approach to factorization.
- **BaselineOnly:** A basic model that accounts for user and item biases without additional complexity.

After training and evaluating each model, the results were as follows:  
| Model         | RMSE   | MAE   |
|---------------|--------|-------|
| **SVD**       | 0.7856 | 0.5889|
| **NMF**       | 0.8712 | 0.6626|
| **BaselineOnly** | 0.8630 | 0.6573|

Among these, the SVD model demonstrated the best performance, with an RMSE of 0.7856 and an MAE of 0.5889. This model was most effective in predicting user ratings and delivering accurate recommendations.  
To enhance the recommendation system, we implemented a hybrid approach that combined collaborative filtering using SVD, content-based filtering based on movie genres, and suggestions of newly added movies that users hadn't rated yet. This hybrid approach allowed us to generate more comprehensive and personalized recommendations, balancing between user preferences, content similarity, and novelty.

## Results

We used RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) as metrics to evaluate model performance. The final model achieved an RMSE of 0.7864 and an MAE of 0.5894, indicating its effectiveness in predicting user ratings and making accurate recommendations. The hybrid approach, which included SVD for collaborative filtering, further enhanced the recommendation quality by incorporating multiple methods to address different user needs and preferences.

## Prediction Function

The final function allows making predictions for new data using the trained model. When a prediction is made, the function processes the input data, applies the recommendation algorithm, and outputs the predicted ratings or recommended movies.

## Final Remarks

This project demonstrates the application of machine learning techniques to build a recommendation system using the MovieLens dataset. Future work may involve incorporating additional features, exploring deep learning approaches, or deploying the recommendation system as a web application for real-time use.




