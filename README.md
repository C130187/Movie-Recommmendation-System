# Final-project-laplace2
Final-project-laplace2 created by GitHub Classroom

IEOR 4571, Columbia University - Personalization Theory and applications \
Course project - Recommender system for movielens dataset\
\
Group members:

1. Jay Shah (js5553@columbia.edu)
2. Chandan Garg (cg3176@columbia.edu)
3. Neeraj Ramkumar (nr2728@columbia.ed)
4. Saloni Gupta (sg3910@columbia.edu)

Steps:
1. Prepare data : We sample ~20,000 users and ~1000 movies based on rating timestamp. Run Exploratory_Data_Analysis_and_Data_Sampling.ipynb to create sample dataset as well EDA
2. Model : We tried 2 architectures and a content based recommendation to address cold-start of new movies:
  * Architecture_1 - Hybrid model of serialized (Singular value Decomposition (SVD), Approximate nearest neighbours (ANN), Neural Collaborative Filtering (NCF) using log loss) and Factorized Machines (FM) using Weighted Approximately Ranked Pairwise (WARP) Ranking Loss - Run /Architecture Pipeline/Architecture_1_(pipeline_0)_.ipynb for architecture 1
  ![Architecture1](https://github.com/jayshah1397/MovieLens-Recommmendation/blob/main/images/Architecture1.png?raw=true)

  * Architecture_2 - Serialized model (SVD,ANN,NCF) - Run /Architecture Pipeline/Architecture_2_(pipeline_0)_.ipynb for architecture 2
  ![Architecture1](https://github.com/jayshah1397/MovieLens-Recommmendation/blob/main/images/Architecture1_part1.png?raw=true)

  * Content Based Recommendation - Used metadata of movies and ratings like genres and reviews to extract vector embeddings using Doc2Vec to identify similar movies. Run - /Content Based Recommendation/Content_Based_Recommendation.ipynb 
3. Evaluate : We evaluated models using 
 * accuracy and utility measures like Normalised Discounted Cummulative Gain (NDCG), Precision-recall at k, mean average precision (MAP), user and catalog coverage and novelty (using Expected Popularity Complement) 

File structure: 

* Architecture Pipeline/Architecture_1_pipeline_0 - Pipeline for architecture 1 - hybrid model of serialized (SVD,ANN,NCF) and FM

* Architecture Pipeline/Architecture_2_pipeline_0 - Pipeline for architecture 0 - Serialized model (SVD,ANN,NCF)

* Models/FM - File with exploration of FM model 

* Models/ANN-NCF - File with EDA for ANN, NCF and ANN-NCF model

* Evaluations - File with Evaluation Metric calculations and plotting

* Content Based Recommendation - Folder with content based recommendation model EDA and the Doc2Vec model used to train content based model

* Exploratory data analysis and data sampling - file with EDA and data sampling

* Baselines - Baselines folder with PMF and Bias baseline notebooks

* ANN - Folder with User-User Similarity and User-Item Similarity exploration
