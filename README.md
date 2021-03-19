# Operationalizing Machine Learning
Student : Amir Haddad - March 2021 .
## Table of contents
   * [Overview](#Overview)
   * [Project Steps](#Project-Steps) 
   * [Future work for improvements](#Comments-and-future-improvements)
   * [Dataset Citation](#Dataset-Citation)
   * [References](#References)

***

## Overview

in This project we build a model and deploy it folowing two methods : 
Deploying a model using AutoML : 
- The first method is to deploy a model with AutoML using Machine Learning Studio folowing the steps :
   - Deploy the best model as a Restfull endpoint .
   - Getting the  the REST API endpoint documentation with Swagger UI .
   - Enabeling the application insights to monitor the model with metrics  .
   - Using Apach benchmark to check the quality of the deployment . 

Deploying a pipeline using Azure SDK : 
- The second method will be the same as the first but using Azure Python SDK to publish a pipeline.

## Dataset 
in this project we used the dataset Benchmark that contains marketing data about individuals.
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe a bank term deposit. 
the predictions are represented by _`column y`_ target feature _`yes`_ or _`no`_.

***
## Project Steps

The key steps of the project are described below:

- **Authentication:**
i used my own azure acount subscription to create a Service Principal account and associate it with my specific workspace "udacity project" folowing the steps :
     - install az , az login 
     -install Azure Machine Learning extension 
     -assign the role to the new Service Principal for the given Workspace, Resource Group and User  
         objectId as an "owner"
     
- **Automated ML Experiment:**
once security is enabled and authentication completed. we procedd with the creation of an experiment using Automated ML, configuring a compute cluster, and using that cluster to run the experiment.

- **Deploying the Best Model:**
Once the run experiment completed , we can obtain the summary of all the models and their metrics  , including all  _Best Model_  _Details_ tab.
Runs are sorted by accuracy and the best model apears in the fist rank and will should be selected for deployment.
the model deployment provides a Restfull Endpoint that can  interract with an HTTP API service to sent (POST) and retrieve (GET) results from the model.

- **Enable Logging:**
 Application Insights is enabled with the log.py script to retrieve logs from the deployed Endpoint.
- **Creating an environement** 
to avoid bugs with packages versions we create a new envirnment to train the model with the two methods 
the python version 3.6.9 installed in the enivronement also the packages : 
 - scikit-learn==0.24.1
  - xgboost==0.90
  - the environement named "automl_env" the dependencies was stored in yml image 'conda_dependencies.yml ' and the environment was registred to our workspace ="udacity-project".
  
- **Swagger Documentation:**
The Swagger UI provides the documentation about our Endpoint . (Data structure and predictions ...) 
we deployed the Swagger localy by :
    -Retreiving the swagger.json (wget command)
        - Executing a bash command swagger.sh (the image kubernatees is downloaded localy on our Doker ) 
        - Execute the serve.py script to expose a local swagger.json file so that a swagger-ui service can             pick it up from localhost.
       - Consulting the content of the swagger-ui  http://localhost:8000/swagger.json


- **Consume Model Endpoints:**
we execute the script  ```endpoint.py``` with  _scoring_uri_ as well as the _key_ for authntication to test the model with two requests 
we obteained two predictions in a array from the Endpoint as an answer as described in the Swagger documentations structure .  

- **Create and Publish a Pipeline:**
the second method of the project will be proceeded with ML AZURE SDK the Jupyter Notebook containes the same steps writen in python .

- **Documentation:**
The documentation includes: 1. the [screencast](https://youtu.be/G2CUqiREr5A) in this video i explained with details all the project steps :
- Automated ML Experiment
- Deployment of the best model
- Enable logging
- Swagger Documentation
- Consume model endpoint
- Create and publish a pipeline.

***


### **Step 2: Automated ML Experiment**

in the second part of this project i launched the SDK using the notebook named "aml-pipelines-with-automated-machine-learning-step" with my own azure account .  


**Registered Datasets:**

![Registred datasets URI](img/Registred datasets URI.png?raw=true "Registred datasets URI")

![Bank marketing data](img/Bank marketing data.png?raw=true "Bank marketing data")

**Creating a new Automated ML run:**

I select the Bank-marketing dataset and in the second screen, I make the following selections:

* Task: _Classification_
* Primary metric: _Accuracy_
* _Explain best model_
* _Exit criterion_: 1 hour in _Job training time (hours)_
* _Max concurrent iterations_: 5. Please note that the number of concurrent operations **MUST** always be less than the maximum number of nodes configured in the cluster.

![ML pipeline run](img/ML pipeline run.png?raw=true "ML pipeline run")

**Experiment is completed**

The experiment runs for about 20 min. and is completed:

![Experiment completed](img/Experiment completed.png?raw=true "Experiment completed")

**Best model**

once the experiment completed ,we have all the metrics of the best selected model:

![Best model pipeline](img/Best model pipeline.png?raw=true "Best model pipeline")


another great feature named data guardtails , we can check all useful informations about potential issues encountred with data quality during the model training . In this case, as you can see the unbalanced data is in Alert status.

![data guardtails ](img/data guardtails.png?raw=true "data guardtails")


### **Step 3: Deploy the Best Model**

The folowing step is about  the deployment of the best selected model .
the best model (Run202)  choosed first in ranking .(voting ensemble) .
I deploy the model with _Authentication_ enabled and using the _Azure Container Instance_ (ACI).

Deploying the best model will provide a resfull endpoint that we can comunicate wiht an HTTP API service to send and recieve (POST-GET) requests.


### **Step 4: Enable Application Insights**

After the deployment of the best model, I can enable _Application Insights_ and be able to retrieve logs:

i run the script logs.py to enable Application Insights

Screenshot of the tab running "Application Insights":

!["Enable Applicaiton insights url](img/Enable Applicaiton insights url.png?raw=true "Enable Applicaiton insights url")

We can see _Failed requests_, _Server response time_, _Server requests_ & _Availability_ graphs in real time.

**Benchmark**

Finaly i used the benshmark.sh bash command to check the health of my endpoint :

![benchmark](img/benchmark.png?raw=true "benchmark")


### **Step 5: Swagger Documentation**

**Swagger** is a set of open-source tools built around the OpenAPI Specification that can help us design, build, document and consume REST APIs. One of the major tools of Swagger is **Swagger UI**, which is used to generate interactive API documentation that lets the users try out the API calls directly in the browser.

In this step, I consume the deployed model using Swagger. Azure provides a _Swagger JSON file_ for deployed models. This file can be found in the _Endpoints_ section, in the deployed model there, which should be the first one on the list. I download this file and save it in the _Swagger_ folder.

I execute the files _swagger.sh_ and _serve.py_. What these two files do essentially is to download and run the latest Swagger container (_swagger.sh_), and start a Python server on port 9000 (_serve.py_).

docker image updated 
![Docker image updated](img/Docker image updated.png?raw=true "Docker image updated")
swagger docmentation 
![Swagger documentation](img/Swagger documentation.png?raw=true "Swagger documentation")
after i executed serve.py to serve on the localhost 
![Swagger documentation serve localhost](img/Swagger documentation serve localhost.png?raw=true "swagger.sh run")





### **Step 6: Consume Model Endpoints**

Once the best model is deployed, I consume its endpoint using the `endpoint.py` script provided where I replace the values of `scoring_uri` and `key` to match the corresponding values that appear in the _Consume_ tab of the endpoint: 

**Consume Model Endpoints: running endpoint.py**

![consume the model endpiont](img/consume the model endpiont.png?raw=true "consume the model endpiont")



### **Step 7: Publish and Consume a Pipeline**

after runing the Notepook 'aml-pipelines-with-automated-machine-learning-step' i published the pipeline under the name : 'Bankmarketing Train'

![published pipeline](img/published pipeline.png?raw=true "published pipeline")

**Bankmarketing dataset with the AutoML module** 

![Bank marketing data](img/Bank marketing data.png?raw=true "Bank marketing data")

**Published Pipeline Overview showing a REST endpoint and an ACTIVE status** 

![Pipeline Restfull endpoint](img/Pipeline Restfull endpoint.png?raw=true "Pipeline Restfull endpoint")



***
## Screen Recording

The screen recording can be found [here](https://youtu.be/0AKGw1YOcXw) and it shows the project in action. More specifically, the screencast demonstrates:

* The working deployed ML model endpoint
* The deployed Pipeline
* Available AutoML Model
* Successful API requests to the endpoint with a JSON payload


***
## Future work for improvements

* As I have pointed out in the 1st project as well, the data is **highly imbalanced**:

![Unbalanced Data](img/Unbalanced Data.png?raw=true "Unbalanced Data")

As i mentionned in the previous project there is am issue with unbalanced data , the model was trained more on positive majority so it would favorite the prediction of positive class .

## Feature Engineering 
In this project i've checked some details in 'Data guardrails' that shows potential issues encountred with data quality during the model . 
i think that in next future porject , one of the best practices is to explore more data structure (distribution) as well as checkin correlation between the features (to avoid intercorelation issue )
and check feature importance 
>> Some Feature engineerign tehniques depending on the data structure :

missing data : imputation technique
outliers : imputation 
High variable magnitude : scaling 
Rare labels : (grouing under umbrealla) should be available in train and test datasets 
High cardinality .
The AutoML did a great job in the research of a robust model however , there is always a way to improve the job by Feature engineering step it could be :
linearisation (Guassian transformation) , discretisation , Standardization ...
we could introduce into our pipeline as a step and see how it goes improving our model accuracy
for example we could build a new pipeline but at this time with the step (Smote) technique to create a syntetic version of observations similar to the minority to tackle the unbalanced data .

https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
project 
https://towardsdatascience.com/how-and-why-to-standardize-your-data-996926c2c832
https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159


- [Tutorial: Train Machine Learning Models with Automated ML Feature of Azure]ML(https://thenewstack.io/tutorial-train-machine-learning-models-with-automated-ml-feature-of-azure-ml/)


## Increasing the training time 
 

Another factor that could improve the model is increasing the training time. This suggestion might be seen as a no-brainer, but it would also increase costs and there must always be a balance between minimum required accuracy and assigned budget.


***
## Dataset Citation

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

***
## References
- Example using Hyperdrive 
 https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-parameter-tuning-with-hyperdrive.ipynb
 - Microsoft docs for model optimization .
 https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py
 
 https://docs.microsoft.com/en-us/answers/questions/248696/using-estimator-or-sciptrunconfig-for-pipeline-wit.html
  https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml
 https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/tune-model-hyperparameters
 - AutoML 
 https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py
 
- [Sklearn - logistic regression ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- randomparametersampling)
https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py
- banditpolicy 
https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py
- Cross validation 
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits

- ONNX
https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx

- Bank Marketing, UCI Dataset: [Original source of data](https://www.kaggle.com/henriqueyamahata/bank-marketing)

- Udacity Nanodegree material
- [App](https://app.diagrams.net/) used for the creation of the Architectural Diagram
- [Prevent overfitting and imbalanced data with automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls)
- [Dr. Ware: Dealing with Imbalanced Data in AutoML](https://www.drware.com/dealing-with-imbalanced-data-in-automl/)
- [Microsoft Tech Community: Dealing with Imbalanced Data in AutoML](https://techcommunity.microsoft.com/t5/azure-ai/dealing-with-imbalanced-data-in-automl/ba-p/1625043)
- A very interesting paper on the imbalanced classes issue: [Analysis of Imbalance Strategies Recommendation using a
Meta-Learning Approach](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_34.pdf)
- [Imbalanced Data : How to handle Imbalanced Classification Problems](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
- [Deep learning vs. machine learning in Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning)
- [A Review of Azure Automated Machine Learning (AutoML)](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406)

- [Feature engineering in machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features)
- [New automated machine learning capabilities in Azure Machine Learning service](https://azure.microsoft.com/en-ca/blog/new-automated-machine-learning-capabilities-in-azure-machine-learning-service/)
- [Data featurization in automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features)
- [What are Azure Machine Learning pipelines?] (https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines)
- [https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines](https://azure.microsoft.com/en-ca/blog/real-time-feature-engineering-for-machine-learning-with-documentdb/)

- [Supported data guardrails](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#supported-data-guardrails)
- [Online Video Cutter](https://online-video-cutter.com/)