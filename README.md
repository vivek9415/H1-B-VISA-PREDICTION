VISA APPROVAL PREDICTION

Using Naive Bayes Classifier

Developed by: Vivek Gupta

Smart Bridge-Remote Summer Internship Program
1. INTRODUCTION
In our project, we aim to predict the outcome of H-1B visa applications that
are filed by many high-skilled foreign nationals every year. We framed the problem as a
classification problem and applied Naive Bayes, SVM in order to output a predicted case status
of the application. The input to our algorithm is the attributes of the applicant which will be
further explained in the following parts.
H-1B is a type of non-immigrant visa in the United States that allows foreign
nationals to work in occupations that require specialized knowledge and a bachelor’s degree or
higher in the specific specialty [1]. This visa requires the applicant to have a job offer from an
employer in the US before they can file an application to the US immigration service (USCIS).
USCIS grants 85,000 H-1B visas every year, even though the number of applicants far exceed
that number [2]. The selection process is claimed to be based on a lottery, hence how the
attributes of the applicants affect the final outcome is unclear. We believe that this prediction
algorithm could be a useful resource both for the future H-1B visa applicants and the employers
who are considering to sponsor them.
a. Overview
To predict the outcome of H-1B visa applications based on the attributes of the
applicant ,several machine learning models like SVM, Naive Bayes can be used. Finally, this
can be integrated to a web appliction.
1.2 Purpose
Our aim from the project is to make use of pandas, matplotlib , & seaborn libraries from
python to extract the libraries for machine learning for the Visa prediction.
Secondly, to learn how to hyper tune the parameters using grid search cross validation for the
NaiveBayes machine learning algorithm.
And in the end, to predict whether the Visa applicant can replay the Visa or not using voting
ensemble techniques of combining the predictions from multiple machine learning algorithms
and withdrawing the conclusions.
2. LITERATURE SURVEY
Data mining is the process of analyzing data from different perspectives and extracting
useful knowledge from it. It is the core of knowledge discovery process. The various steps
involved in extracting knowledge from raw data as depicted in figure-1. Different data mining
techniques include classification, clustering, association rule mining, prediction and sequential
patterns, neural networks, regression etc. Classification is the most commonly applied data
mining technique, which employs a set of preclassified examples to develop a model that can
classify the population of records at large. Fraud detection and credit risk applications are
particularly well suited to classification technique. This approach frequently employs Decision
tree based classification Algorithm. In classification, a training set is used to build the model as
the classifier which can classify the data items into its appropriate classes.
A test set is used to validate the model.
2.1 Proposed Solution
Machine Learning (Naive Bayes):
Naive Bayes algorithm in machine learning methods which efficiently performs both
classification and regression tasks. Naive Bayes is a kind of classifier which uses the Bayes
Theorem. It predicts membership probabilities for each class such as the probability that given
record or data point belongs to a particular class. The class with the highest probability is
considered as the most likely class. And the mot likely class will be the output predicted for the
loan estimation.
And also we have created an UI using the Flask for the loan status prediction, this UI will
allow the users to predict the loan status very easily and the User interface is user friendly not at
least one complication in using the interface, and it can be used just by entering some necessary
details into the UI in real time it'll give the predicted value like if the customer is beneficial to
take a loan and how often does he pays the loan interest amount to the bank.
Basically this model will give the predicted value when a customer with details will pay
the loan back to bank, by just taking some necessary details of the customer in real time, and
those details will be collected by bank employee within minutes.
3. THEORETICAL ANALYSIS
While selecting the algorithm that gives an accurate prediction we gone through lot of
algorithms which gives the results abruptly accurate and from them we selected only one
algorithm for the prediction problem that is Naive Bayes Classifier, it assumes that the presence
of a particular feature in a class is unrelated to the presence of any other feature.
thats how the prediction work great with the Naive Bayes Algorithm.
The peculiarity of this problem is collecting the customers details real time and working
with the prediction at the same time, so we developed an user interface for the people who'll be
accesssing for the Visa status prediction. Accuracy is defined as the ratio of the number of
samples correctly classified by the classifier to the total number of samples for a given test data
set. The formula is as follows
Accuracy=TP+TN/TP+TN+FT+FN
At first we got like lot of worst accuracies because we tried lot of algorithms for the best
accurate algorithm , finally after all of that we tried the best suitable algorithm which gives the
prediction accurately is Naive Bayes Classifier. And developed it to use as a real time prediction
probelm for the visa status prediction.
3.2 Software Designing
1. Jupyter Notebook Environment
2. Spyder Ide
3. Machine Learning Algorithms
4. Python (pandas, numpy, matplotlib, seaborn, sklearn)
5. HTML
6. Flask
We developed this Visa Approval status prediction by using the Python language which is
a interpreted and high level programming language and using the Machine Learning algorithms.
for coding we used the Jupyter Notebook environment of the Anaconda distributions and the
Spyder, it is an integrated scientific programming in the python language.
For creating an user interface for the prediction we used the Flask. It is a micro web
framework written in Python. It is classified as a micro frame work because it does not require
particular tools or libraries. It has no database abstraction layer, form validation, or any other
components where pre-existing third-party libraries provide common functions, and a scripting
language to create a webpage is HTML by creating the templates to use in th functions of the
Flask and HTML.
4. EXPERIMENTAL INVESTIGATION
In this paper, the dataset we used is derived from H-1B_Kaggle .It contains more than
10L H-1B Visa data of users .It contained 7 features and 1 label which can be examined
attributes. Those attributes were shown below in the screenshot of the data set we used.
CASE STATUS: We excluded the cases ’CERTIFIED-WITHDRAWN’ and ’WITHDRAWN’,
since ’WITHDRAWN’ decisions are either made by the petitioning employer or the applicant,
therefore not predictive of USCIS’s future behavior. We labeled ’CERTIFIED’ cases as 1 and
’DENIED’ cases as 0.
FULL TIME POSITION: Positions are given in ”Full Time Position = Y; Part Time Position =
N” format. We converted them to ”Full Time Position = 1; Part Time Position = 0” format.
YEAR: Year in which application was filed. We converted the data into one-hot-k representation.
PREVAILING WAGE: Prevailing wage is the average wage paid to employees with similar
qualifications in the intended area of employment. We discarded the outlier terms and used the
rest of the data as it was. APPS PER
EMPLOYER_NAME: We created a feature for the number of H-1B applications per employer,
and discarded data points that are petitioned by an employer that has less than 4 applications.
Although this processing step undesirably gets rid of applications filed by small companies, it
significantly helps with cleaning up the misspelled company names.
We created a feature for the success rate per employer. APPS PER
SOC_NAME: SOC stands for Standard Occupational Classification System, which is a federal
occupational classification system. We created a feature for the number of H-1B applications per
SOC type, and discarded data points with SOC types that appear less than 4 times in the data.
This processing step undesirably gets rid of applications with uncommon jobs, but helps with
cleaning up .
WORKSITE: Data is given in the ”City, State” format. We only included ”State” and converted
the data
into one-hot-k representation.
After the pre-processing steps described above, we split the training, Training set had a total of
1.2 million examples. Due to the inherent bias in our dataset towards the ”CERTIFIED” label,
we created two versions and test sets in order to make sense of the error analysis later on. First
version of dev and test sets were both unbalanced, each consisting of 400K examples. More
specifically, around 90% of the examples had a ”CERTIFIED” label, mimicking the nature of the
original dataset. Second version of dev and test sets were manually balanced by sampling
”CERTIFIED” labeled examples roughly equal to the number of ”DENIED” labeled examples.
5. Process Flow of Project
6. FLOW CHART
7. RESULT
In this paper, the Naive Bayes algorithm is used to predict its performance, and compared
with another machine learning methods namely the decision tree, the logistic regression and the
SVM. The obtained results are displayed in Table below. The results show that, the performance
of Naive Bayes have comparable performance than that of logistic regression, random forest,
SVM and decision tree, but the Naive Bayes still performs the best, with an accuracy of 98%,
8. ADVANTAGES AND DISADVANTAGES
Advantages:
1. Naive Bayes give the accurate result of the prediction upto 98% which is the algorithm
we used for prediction.
2. H-1B visa benefit,and perhaps the main reason for its popularity, is the board
requirements associated with qualifying for the visa
3. Duration of Stay
4. Portability
5. Anyone Can Apply
6. Dual Intent (pursue legal permanent residency) while under H-1B non-immigrant status.
Disadvantages:
1. Lottery.
2. Extensions.
3. Due to lottery process,there are strict dates the must be adhgered to during process.
4. Fees.
9. CONCLUSION
Inorder to predict the outcome of H-1B visa applications based on the attributes of the
applicant ,several machine learning models like SVM, Naive Bayes can be used.
Finally,this can be integrated to a web appliction.
10. FUTURE SCOPE
In further Naive Bayes algorithm can be applied on other data sets available for visa approvals to
further investigate its accuracy. A rigorous analysis of other machine learning algorithms other
than these six can also be done in future to investigate the power of machine learning algorithms
for visa status prediction. In further study, we will try to conduct experiments on larger data sets
or try to tune the model so as to achieve the state -of-art performance of the model and a great
UI support system making it complete web application model.
