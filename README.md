# Diabetes-classification
This a data science project related to diabetic patients

The dataset in this project is related to the information gathered from diabetic patients in 130 hospitals around the US between the 1999 to 2008 years. This dataset contains more than 100,000 records and 50 attributes. The last column “readmitted” indicates whether a diabetic patient needed hospitalization again after discharge. The purpose of this project is to predict the “readmitted” feature, whether a patient needed hospitalization after discharge or not, based on the information provided by other attributes. In the “readmitted” column, ‘No’ means the patient didn’t need hospitalization after discharge, ‘<30’ means the patient needed hospitalization in less than 30 days after discharge, and ‘>30’ means the patient needed hospitalization more than 30 days after discharge. So, we should separate the patients into 3 classes. 
A detailed explanation about all attributes is available on the link below:
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

The project has been divided into 4 steps:
1-	Handling missing values
2-	Feature selection
3-	Handling outliers
4-	Model deployment

•	Handling missing values: Based on the conditions of the corresponding attribute as well as the number of missing values, missing values of each attribute have been treated differently. 
o	For the “race” attribute, missing values have been replaced by the most frequent value in their corresponding column (Caucasian).
o	For the “age” attribute, the center of each interval is replaced by that interval.
o	For the “weight” attribute, the mean of all known values is replaced by missing values.
o	For “gender”, “diag_1”, “diag_2”, and “diag_3” attributes, the rows with missing values are deleted. That won’t delete many records from our dataset because the number of missing values in these features are very small.
o	The “payer_code” attribute will be entirely omitted from our dataset because it contains so many missing values and therefore doesn’t provide any valuable information.
o	All other missing values in the remaining attributes are left to be missed, because their number is relatively large and because of the properties of their attribute, we cannot suggest any replacing value for them. We cannot either decide to omit their attributes because they may contain some valuable information.
All of the above mentioned procedures are implemented in the “EliminateMissingValues()” function in the “preprocessing.py” file. After calling the mentioned function, a CSV file that all of its missing values are handled will be generated. (diabetic_data_without_missing_values.csv)

•	Feature Selection: We consider two criteria for selecting features. Mutual Information, and how well a given attribute separates the data (for categorical attributes). For the first criteria, we calculate all of the attributes’ mutual information with respect to the “readmitted” attribute. These quantities are saved in the “mutual_informatin.pkl” file. For the second criteria, we separate the data based on the values they have for a given attribute. Then, we measure the purity of each group. Consequently, we use GINI_Split for each attribute. Then, we select attributes with higher GINI_Split and Mutual information. 
After performing the above mentioned procedures, we ended up selecting 11 categorical attributes and 7 numerical ones. However, after model deployment, we realized that values of numerical attributes are not distributed normally (which ruins the model’s performance) and the information provided by the selected categorical attributes is enough for making an accurate classification. Therefore, we decided to ignore numerical attributes all together. The selected features are written in a list in the “preprocessing.py” file. The “FeatureSelection()” function rules out unselected attributes and generates a file named “selected_features.csv”. This file only contains the selected attributes.

•	Handling outliers: In this part, the records that have values which the probability of their occurrence is less than a determined threshold, are deleted. For instance, only one person out of 100224 has the ‘V51’ value for the “diag_1” attribute. So, his corresponding record will be deleted. The outlier threshold is set to 0.0001 (0.01%). This threshold has been determined empirically. The “DetectOutlier()” function in the “helper_functions.py” file is responsible for recognizing outliers. The “EliminateOutliers()” function will rule out the detected noises. This function generates the “preprocessed_data.csv” file which doesn’t have any noises and will be used for model deployment.
After performing the preprocessing, there will be 96227 records in the dataset.

•	Model Deployment: According to the previous research, this dataset is suitable for statistical analysis. Therefore, we apply a statistical model for classification. At first, the simplest statistical model Naïve Bayes was implemented and tasted. Fortunately, based on the 10-fold-cross-validation testing, Naïve Bayes has a 99.74% accuracy in predicting the “readmitted” attribute.
