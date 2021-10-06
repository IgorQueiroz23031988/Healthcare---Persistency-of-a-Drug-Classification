# Healthcare - Persistency of a DRug - Classification
Machine Learning classification model to classify if futures patients will take the drugs prescribed by the doctor during the entire treatment or if they won't.

Link for the streamlit app: <https://share.streamlit.io/igorqueiroz23031988/healthcare-persistency-of-a-drug-classification/main/deploy2.py>

![alt text](https://github.com/IgorQueiroz23031988/House-Sales-Data-Anaysis-and-Insights/blob/main/HOUSE%E2%80%99S%20BUY%20AND%20SALES%20DATA%20ANALYSIS%20AND%20INSIGHTS.png)

## 1. Business Description/Problem.

One of the challenges for all Pharmaceutical companies is to understand the persistency of drug as per the physician prescription. To solve this problem ABC pharma company approached an analytics company to automate this process of identification.

 With an objective to gather insights on the factors that are impacting the persistency, it is necessary to build a classification for the given dataset, using the variable ‘Persistency_Flag’ as target variable and other attributes as prediction variables.

## 2. Business Understanding.

ABC it is a private pharma company. Due to the problem to the persistency of drug as per the physician prescription, a data science project is applied to predict the classification of ‘Persistency_Flag’ variable. In other words, based on the previously patients characteristics it is possible predict if futures patients will use the drugs during the role treatment or if they won’t. 

The object of this project is providing answer of the main questions made by the company’s CEO, which are:

*	What is the ‘Persistency_Flag’ classification for future patients?

The answer for those questions is presented in two different methods:

*	A webapp with all necessary prediction attributes in order to predict the classification of the ‘Persistency_Flag’ for future patients. 

*	A dashboard with several hypotheses and insights to help the company CEO with future decisions.

The tools used for this project are: Python 3.8, Pycharm, Jupyter Notebook, Streamlit and Heroku.

## 3. Data Understanding.

There is 1 dataset provided: <https://www.kaggle.com/harlfoxem/housesalesprediction>.

 Variables Description:
 
Here I'm describing the columns in detail:

Patient Details:

*	Patient ID: Unique ID of each patient;
*	Persistency_Flag: Flag indicating if a patient was persistent or not;
*	Age: Age of the patient during their therapy;
*	Race: Race of the patient from the patient table;
*	Region: Region of the patient from the patient table;
*	Ethnicity: Ethnicity of the patient from the patient table;
*	Gender: Gender of the patient from the patient table;
*	IDN Indicator: Flag indicating patients mapped to IDN;

Provider Attributes:

*	NTM - Physician Specialty: Specialty of the HCP that prescribed the NTM Rx;

Clinical Factors:

*	NTM - T-Score: T Score of the patient at the time of the NTM Rx (within 2 years prior from rxdate);
*	Change in T Score: Change in Tscore before starting with any therapy and after receiving therapy (Worsened, Remained Same, Improved, Unknown);
*	NTM - Risk Segment: Risk Segment of the patient at the time of the NTM Rx (within 2 years days prior from rxdate);
*	Change in Risk Segment: Change in Risk Segment before starting with any therapy and after receiving therapy (Worsened, Remained Same, Improved, Unknown);
*	NTM - Multiple Risk Factors: Flag indicating if patient falls under multiple risk category (having more than 1 risk) at the time of the NTM Rx (within 365 days prior from rxdate);
*	NTM - Dexa Scan Frequency: Number of DEXA scans taken prior to the first NTM Rx date (within 365 days prior from rxdate);
*	NTM - Dexa Scan Recency: Flag indicating the presence of Dexa Scan before the NTM Rx (within 2 years prior from rxdate or between their first Rx and Switched Rx; whichever is smaller and applicable);
*	Dexa During Therapy: Flag indicating if the patient had a Dexa Scan during their first continuous therapy;
*	NTM - Fragility Fracture Recency: Flag indicating if the patient had a recent fragility fracture (within 365 days prior from rxdate);
*	Fragility Fracture During Therapy: Flag indicating if the patient had fragility fracture during their first continuous therapy;
*	NTM - Glucocorticoid Recency: Flag indicating usage of Glucocorticoids (>=7.5mg strength) in the one year look-back from the first NTM Rx;
*	Glucocorticoid During Therapy: Flag indicating if the patient had a Glucocorticoid usage during the first continuous therapy;

Disease/Treatment Factors:

*	NTM - Injectable Experience: Flag indicating any injectable drug usage in the recent 12 months before the NTM OP Rx;
*	NTM - Risk Factors: Risk Factors that the patient is falling into. For chronic Risk Factors complete lookback to be applied and for non-chronic Risk Factors, one year lookback from the date of first OP Rx;
*	NTM - Comorbidity: Comorbidities are divided into two main categories - Acute and chronic, based on the ICD codes. For chronic disease we are taking complete look back from the first Rx date of NTM therapy and for acute diseases, time period before the NTM OP Rx with one year lookback has been applied;
*	NTM - Concomitancy: Concomitant drugs recorded prior to starting with a therapy (within 365 days prior from first rxdate)
Adherence: Adherence for the therapies.


## 3. Business Assumptions.

After carefully research, some assumptions are taken based on several information obtained at website <https://www.kaggle.com/harlfoxem/housesalesprediction/discussion>.
 
Those assumptions lead to identify possible outliers’ existent on dataset. Such as:
 
*	Any house which contains no bathrooms or bedrooms is considered outlier, therefore it is excluded.
*	Any house which the number of bedrooms is higher than 11 is considered outlier, therefore it is excluded.

Furthermore, some assumptions were made to identify the profit range by selling houses.
 
*	According the website: <https://www.prnewswire.com/news-releases/average-us-home-seller-profits-hit-65-500-in-2019--another-new-high-300991828.html>, the minimal profit made by selling houses In US is 10%, the maximum is 45%. This profit range flouts due the houses characteristics, such as: location, size, number of bedrooms and others. 

## 4. Solution Strategy.

Solution adopted to generate business insights and create a ML classification model to solve the proposed problem. This solution includes:

* Data Description;

1º - Data Dimensions.
 
2º - Descriptive Statistics.   
 
3º - Find the house’s price median by region.
 
4º - Recommend that the houses with prices inferior to the median value should be bought, and the condition is minimal 3.  
 
5º - Filter those houses, that should be bought, by size, number of floors, number of bedrooms, and number of bathrooms, in order to identify the level of recommendation of each house.<br/><br/>
 
 
* __Second part:__ When to sell the houses and for how much?
 
__1º__ - After the company buys the houses, the data is grouped by region and seasons.
 
__2º__ - Inside each region and seasons, it is calculated the median price.   
 
__3º__ - If the buy price is higher than median price plus season and recommendation to buy is regular, than the sell price will be equal the buy price plus 10 %. 
 
   If the buy price is higher than median price plus season and recommendation to buy is high, than the sell price will be equal the buy price plus 12.5 %.
 
   If the buy price is higher than median price plus season and recommendation to buy is very high, than the sell price will be equal the buy price plus 15 %.
 
   If the buy price is lower than median price plus season and recommendation to buy is regular, than the sell price will be equal the buy price plus 30 %.
 
   If the buy price is lower than median price plus season and recommendation to buy is high, than the sell price will be equal the buy price plus 37.5 %.
 
   If the buy price is lower than median price plus season and recommendation to buy is very high, than the sell price will be equal the buy price plus 45 %.
 
__4º__ - It is specified the best moment to sell based on the profit by season.
 
__5º__ - It is specified the best moment to sell based on the profit by season and recommendation to buy in general and individual houses.
 
__6º__ - It is specified the total profit by buying and selling houses.
 
## 5. Top 08 Data Insights.

__Hypothesis 01:__ Houses which has water view are 20% more expensive, in general.

__False:__ Houses with water view are 212.57668803323867 percent more expensive.<br/><br/>


__Hypothesis 02:__ Houses that was built before 1955 are 50% cheaper, in general.

__False:__ Houses that was built before 1955 are -0.7757205525248732 percent cheaper.<br/><br/>


__Hypothesis 03:__ Houses without basement are 40% bigger them house with basement, related to total area (sqft_lot). 

__False:__ Houses without basement are 22.483151526642544 percent bigger them houses with basement.<br/><br/>


__Hypothesis 04:__ The growth of house prices YoY (Year over Year) (May 2014 compared to May 2015) is 10%, in general. 

__False:__ The total houses price YoY (Year over Year) suffered a decrease of -62.79177358882806 percent.<br/><br/>


__Hypothesis 05:__ Houses with 3 bathrooms have a growth MoM (month over Month) of 15%.

__False:__ The total houses price MoM (month over Month) suffered a decrease of -9.953899240174858 percent.<br/><br/>


__Hypothesis 06:__ Houses with number of bedrooms above 8 have a number of bathrooms 40% higher than houses with number of bedrooms between 5 and 8, and 94% higher than houses with number of bedrooms between 1 and, 4 on average.

__True:__ Houses with number of bedrooms above 8 have a number of bathrooms 39.9514563106796 percent higher than houses with number of bedrooms between 5 and 8, and 94.48676155875182 higher than houses with number of bedrooms between 1 and 4.<br/><br/>

__Hypothesis 07:__ Houses with 7 bedrooms has the total area (sqft_lot) bigger between 132 to 320 percent than houses with 8 to 11 bedrooms, on average.

__True:__ Houses with 7 bedrooms has the total area (sqft_lot) bigger between 132.29431644290653 and 320.17243208828523 percent than houses with 8 to 11 bedrooms.<br/><br/>


__Hypothesis 08:__ Renovated Houses have living rooms 12% bigger than houses not renovated, on average.

__True:__ Renovated Houses have living rooms 12.132344286788795 percent bigger than houses not renovated, on average.

## 6. Financial Results.

 House Rocket Company would have a profit of almost 19 percent, which are more than $771 million, if applies this data analytics method.

## 7. Conclusion.

In conclusion, it is possible to identify that the application of data analytics project at dataset from House Rocket Company was very successful, providing a huge profit opportunity based on which houses to buy and when to sell.

## 8. Next Steps.

Other project that can be made with this dataset is the exploration data analyses, which identify the best’s attributes in order to apply machine learning algorithms, with the objective to predict the price of futures houses to buy.



