## Cloud Computing for Data Analysis: Rating Prediction System

<img src="imagePath/imageFile.jpg" class = "inline"/>

#### Authors:
* Akshay Bankar (abankar1@uncc.edu) 
* Keyukumar Hansoti (khansoti@uncc.edu)
* Kiran Korey	(kkorey@uncc.edu)
***

#### Overview:
* The Project aims to predict ratings based on the reviews provided by the users.
* The goal of the project is to implement k-Nearest Neighbors and Naive Bayes Algorithm with the help of PySpark.
***

#### Motivation:
* Every single product or place today has its own profile on web giving out information about it to the customers and visitors.
* It not only tells the user about its features but also the reviews and ratings given to it by the people who have experienced it.
* This urged us to automate the technique of rating prediction based on reviews and learn about the algorithms behind it.
* The goal of this project is to determine whether a customer should buy that product or not just by looking at the ratings. 
* This project can be used to any data set which involves user reviews and can predict its ratings though, some pre-processing could be required.
***
#### Data:
* The Kaggle dataset “Hotel Reviews Data in Europe” will be used which is provided by the booking.com.
* The data set contains 515,000 customer reviews and scoring of 1493 luxury hotels across Europe.
* The data set contains 17 fields like Hotel Name, Average Score, Negative Review, Positive Review, Review Total Negative Word Counts, Review Total Positive Word Counts, Reviewer Score.
* Out of the 17 fields, we will be using Negative Review, Positive Review and Reviewer Score fields.
    - Negative Review: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
    - Positive Review: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive' 
    -  Reviewer Score: Score the reviewer has given to the hotel, based on his/her experience.
* The Negative Review and Positive Review will be used as the features and Reviewer Score will be used as targets. The reviews will be converted to count vectors and fed to the model.

* ##### Data Link: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)
***


#### Tasks Involved and Steps Implemented:
* Understanding the algorithm
* Setup a cluster using  Amazon EMR with spark.
* Install necessary software like Jupyter Notebook, sci-kit learn library, nltk library, pandas, etc..
* Implementing k-Nearest Neighbors and Naive Bayes with sci-kit learn Library.
* Deploying the code and data on Amazon EMR.
* Output generation.
* Implementing k-Nearest Neighbors and Naive Bayes with pyspark.
* Execute the algorithms and generate reports.
* Project Report
***

#### Algorithms :

### K-Nearest Neighbor: 
K-nearest neighbour clustering (KNN) is a supervised classification technique that looks at the nearest neighbours, in a training set of classified instances, of an unclassified instance in order to identify the class to which it belongs.

The model for KNN is the entire training dataset. When a prediction is required for a unseen data instance, the kNN algorithm will search through the training dataset for the k-most similar instances. 
The prediction attribute of the most similar instances is summarized and returned as the prediction for the unseen instance.

The similarity measure is dependent on the type of data. For real-valued data, the Euclidean distance can be used. 
Other other types of data such as categorical or binary data, Hamming distance can be used.

In the case of regression problems, the average of the predicted attribute may be returned. 
In the case of classification, the most prevalent class may be returned.

The KNN algorithm belongs to the family of instance-based, competitive learning and lazy learning algorithms.

Instance-based algorithms are those algorithms that model the problem using data instances (or rows) in order to make predictive decisions. 
The KNN algorithm is an extreme form of instance-based methods because all training observations are retained as part of the model.

It is a competitive learning algorithm, because it internally uses competition between model elements (data instances) in order to make a predictive decision. 
The objective similarity measure between data instances causes each data instance to compete to “win” or be most similar to a given unseen data instance and contribute to a prediction.

Lazy learning refers to the fact that the algorithm does not build a model until the time that a prediction is required. 
It is lazy because it only does work at the last second. This has the benefit of only including data relevant to the unseen data, called a localized model. 
A disadvantage is that it can be computationally expensive to repeat the same or similar searches over larger training datasets.

---
 > **Procedure**  
---
* Load data (here data is loaded from a file)
* Pre process the data using NLP techniques 
    * Tokenize the reviews and remove stop words, emoticons, punctuations, non alphanumeric words
    * Perform stemming on the tokens using Porter Stemmer.
    * Perform lemmatizing on the tokens using WordNetLemmatizer.  
    * Convert class labels to int
* Convert the reviews to sparse matrix of Count vectors
    * We have used Sci-kit library CountVectorizer to convert the reviews to sparse matrix.
    * we have ignored the words who appear in more than 70% of the documents and have considered only the top 500 features to reduce the feature space.
* Split the data into test and training sets
* Compute the distance from a test instance (A member of the test set) to all members of the training set
    * Euclidean distance is used as distance measure.
    * Which is the Square root of sum of squares of the difference between the values of the instances.
* Select the K nearest neighbours of the test instance
* Assign the test instance to the class most represented in these nearest neighbours

> For this algorithm the distance between every instance of the test set has to calculated with every instance of test set, for this we tried using cartesian method, but this causes memory error and also takes a lot of time.
So, we changed the approach to collect each test instance and map each train instance to the collected test instance.  
---
### Naive Bayes
Naive Bayes is a classification algorithm for binary (two-class) and multi-class classification problems. It is called naive Bayes or idiot Bayes because the calculation of the probabilities for each hypothesis are simplified to make their calculation tractable. Rather than attempting to calculate the values of each attribute value P(d1, d2, d3|h), they are assumed to be conditionally independent given the target value and calculated as P(d1|h) * P(d2|H) and so on.

> Steps

***

#### Expectations/Aspects
##### 1. What to expect?

The main aim of the project deliverable will be to predict the rating based on the reviews provided by the users.

 1.  Data will be pre-processed using NLP techniques like Stemming and Lemmatization to make sense of the reviews provided by the customer and how harsh can that review be on the rating.
 2. After pre processing the Negative Review column and Positive Review column we will combine them to make a single column “Review” which will be used as a input for the algorithms.
 3. The data will be divided into training and test set and use training set to train the models and use test set to calculate the accuracy of the models based on the unseen data.
 4. Implementation of both the algorithms and a proper documentation of outcomes of this project, which is the prediction of rating for customers reviews.
 5. Compute the accuracy of the models.
 6. Run the models in Amazon EMR

##### 2. Likely to accomplish


1. Compare the results of KNN and Naïve-Bayes implemented algorithms.
2. Implementation of KNN model with sci-kit libraries and comparison of the accuracy
3. Implementation of Naïve Bayes model with sci-kit libraries and comparison of the accuracy
4. Documentation and online-publishing of the codebase.

##### 3. Ideal Accomplishments.

1. Compare the results of the implemented models with the library implementations like sci-kit learn
2. Suggested modifications/changes in the existing or project implementation.
3. ~~Perform K-Fold cross validation and tune the algorithms by testing it on different alpha values for Naïve Bayes and K values for KNN algorithms.~~
4. ~~Creating a easy to use library that one can use for analysis purpose.~~
***


#### Tools & Technology

* Amazon EMR (Spark 2.2.0) (1 master and 2 slaves) for running the program on cluster.
* Apache Spark - pySpark
* Jupyter Notebook
* Git for tracking the code changes.
* GitHub for hosting the website.

***
####  Installation

1.  Create a Spark Cluster on Amazon EMR and get the details of the cluster to use it on Terminal.
	 Choose the following configuration for the cluster: 
	~~~~
	Spark: Spark 2.3.2 on Hadoop 2.8.5 YARN with Ganglia 3.7.2 and Zeppelin 0.8.0
	~~~~
2. Connect to the Cluster with the keypair you selected.
	You might need to add SHH rule to the Master EC2 Security group
	~~~~
	ssh -i keypair.pem  hadoop@ec2-52-91-121-171.compute-1.amazonaws.com
	~~~~
3. Install / Import following packages to the cluster.
	~~~~
	sudo pip install jupyter
	sudo  pip install sklearn
	sudo pip install pandas
	sudo pip install -U nltk
	~~~~
	Open python and execute the following code to download the nltk data.
	~~~~
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")
	~~~~
4. Repeat Step 2 and 3 to the Slave EC2 instances.
	In our case we had 2 slave EC2 instances. 
	To connect to Slaves you have to login as **ec2-user**
	~~~~
	ssh -i keypair.pem ec2-user@ec2-54-197-45-104.compute-1.amazonaws.com
	~~~~
	> For nltk you have to move the **nltk_data** to "**/home/**" location
5. 
* Run PCSalgorithm on input data consisting 4,20,000 ratings stored on S3 Storage. To recommend movies for user 1199825.
~~~~
spark-submit s3://itcs6190/PCSalgorithm.py s3://itcs6190/movie_input_ratings.txt s3://itcs6190/movie_titles.csv 1199825
~~~~

* Run ALS on input data consisting 4,20,000 ratings stored on S3 Storage. To recommend movies for user 1199825.
~~~~
spark-submit s3://itcs6190/ALS.py s3://itcs6190/movie_input_ratings.txt s3://itcs6190/movie_titles.csv 1199825
~~~~

* Run ALS using library for recommend movies for all users
~~~~
spark-submit s3://itcs6190/ALSUsingLibrary.py s3://itcs6190/movie_input_ratings.txt
~~~~

***

#### Outputs for User ID: 1488844 & Results

The programs to recommend were ran on Amazon EC2 Spark cluster. And satisfactory recommendations were obtained using 3 methods.

* k-Nearest Neighbor implementation.

<img src="images/Emoticons/PCS.png" alt ="Pearson Correlation Coefficient Recommendation" class = "inline"/>

* Naive Bayes implementation.

<img src="images/Emoticons/ALS.png" alt ="Alternating Least Squares Recommendation" class = "inline"/>


***

#### Conclusion:

* Statement1
* Statement2

***

#### Future Scope:

* Statement1
* Statement2 

***

#### Code Snippet:

* Data Cleaning

<img src="imagePath/imageFile.png" class = "inline"/>

* kNN

<img src="imagePath/imageFile.png" class = "inline"/>

***

#### Challenges Faced:

* Statement1
* Statement2

***

#### Work Division:

The complete project has been accomplished together with inputs from both the team members. 


| Number  | Task                  | Contribution      |
| ------- :|:---------------------:| -----------------:|
| 1       |  Task1		  |        MemberName |
| 2       |  Task2		  |        MemberName |
| 3       |  Task3		  |        MemberName |
| 4       |  Task4 		  |        MemberName |
| 5       |  Task5		  |        MemberName |

***

#### References:
*

*

*

*
