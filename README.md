# Acme Bank Corporation Churn Predict :classical_building: :euro: :credit_card: :dollar:
![GitHub last commit](https://img.shields.io/github/last-commit/MEziliano/Bank-Churn-Predict?style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/MEziliano/Bank-Churn-Predict?style=for-the-badge)
[![GitHub issues](https://img.shields.io/github/issues/MEziliano/Bank-Churn-Predict?style=for-the-badge)](https://github.com/MEziliano/Bank-Churn-Predict/issues)
![Badge em Desenvolvimento](https://img.shields.io/static/v1?label=STATUS&message=FINISHED&color=CYAN&style=for-the-badge)


<h2> Introduction </h2> 

The loyalty of costumers is an important asset in the revenue of any company, and maybe this is one of the greatest fields of Data Science. The point is simple: know if a customer will leave or not the comapny services. 
Thar happens for a few reasons:

* It harder to acquire new clients than to keep the existing ones.

* The chance that a client who left the financial institution returns in the future is derisory.

* A client that left the institution is less likely to reccommend it for other people, becoming a detractor of the services of the company.



The goal of this project is try to predict if a customer will quit or not quit. And for this task we will analyze a dataset with features about customers of ACME Bank Corporation, as we can see below. 

<details><summary><h3>Data Dictionary</h3></summary>
<p>

| Column  | Description | Data Type
| ------------- | ------------- | ------------- | 
| CustomrtId            | The customer unique identifying number | id |
| Surname               | The customer surname | string type|
| CreditScore           | The customer credit rank in the bank | Continuous variable|
| Geography             | Residence by country | Discrete variable|
| Gender                | The customer gender| Binary category as string type|
| Age                   | age in years| continnuos variable|
| Tenure                | The number of customer possessions| discrete variable|
| Balance               | Account balance| numerical continuos |
| NumOfProducts         | The number of financial products used by the customer| numerical discrete variable|
| HasCard               | Has or not credit card| binary variable|
| IsActiveMember        | Indicates if the costumer is active or not| binary variable|
| EstimatedSalary       | Estimated Salary| continuous variable|
| Exited                | Costumers who get out the service | Target in classification model |
 

 </p>
</details>

<h2> Exploratory Data Analysis — EDA </h2>

From ten thounsand costumers records in this dataset is possible to check that 20.4% had left the bank services. According to some analyzes performed, it's possible to check the countrie with most churning rate was Germany. Also it was checked the proportion of genders and others features, but an interesting observation is: every costumer with all four services had left the bank.  


At the part of the Exploratory Data Analysis was possible check which features could be splited to the better performance of the Machine Learning. 

<h2> Machine Learning </h2>

At the part of the Machine Learning a several models was trained, after that was possible to pick one and choose the best parameter aiming at the best performance according with the metric of accuracy, but other metrics were taken into account.

<h3> Hyperparameter</h3>
After all, it was necessary to improve the best model and made the deploy. But, with this big dataset, it's was necessary to develop a few lines of code and work with every parameter separately. Check the code!   

-----------------------------

<h3> Used in the project! </h3>

<div>
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" target="_blank">
<a href="https://www.kaggle.com/sidneyviana/customer-churn-classifier/notebook"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white"></a>
<a href="https://colab.research.google.com/drive/1_1wbhW2zD1JjxQmQyqFxX3gu_GSOZ8R7"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" alt="Open In Colab"/ target="_blank"></a> 
<ahref><img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" target="_blank">
<img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" target="_blank">
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" target="_blank">
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" target="_blank"> 

</div>
<div>
<h3> Check also this comments</h3>
<a href="https://medium.com/@murilosez06/a-week-inside-a-data-science-project-eabcfd2a2c56" target="_blank"><img align="center" src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" target="_blank"></a>
<a href="https://www.notion.so/muriloeziliano/Classification-d621168874bf435780c6b63196e4c8cd" target="_blank"><img align="center" src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white"></a>
</div> 

