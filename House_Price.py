import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model,svm
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt


train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv") #Load train data (Write train.csv directory)
test = pd.read_csv("house-prices-advanced-regression-techniques/test.csv") #Load test data (Write test.csv directory)

data = train.append(test,sort=False) #Make train set and test set in the same data set



data = data.fillna(0)
data = pd.get_dummies(data)




norm_value = 755000

#Drop features that are correlated to each other

#Yo chai multiple feature bich reln check garna, if same nai effect garxa vani multiple feature rakhnu watse hunxa instead aautai rakhda hunxa
covarianceMatrix = data.corr()


#yo chai list of features
listOfFeatures = [i for i in covarianceMatrix]

setOfDroppedFeatures = set() 
for i in range(len(listOfFeatures)) :
    for j in range(i+1,len(listOfFeatures)): #Avoid repetitions 
        feature1=listOfFeatures[i]
        feature2=listOfFeatures[j]
        if abs(covarianceMatrix[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8
            setOfDroppedFeatures.add(feature1) #Add one of them to the set
#I tried different values of threshold and 0.8 was the one that gave the best results

print("Drop garna milni columns",setOfDroppedFeatures)
data = data.drop(setOfDroppedFeatures, axis=1)


#ani output snaga reln nai navako..i.e kam effect garni ni faldida hunxa

#Drop features that are not correlated with output

nonCorrelatedWithOutput = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]
#I tried different values of threshold and 0.045 was the one that gave the best results

print("\n\n\n\n")
print("Output ma effect nagarni",nonCorrelatedWithOutput)
data = data.drop(nonCorrelatedWithOutput, axis=1)


#First, we need to seperate the data (Because removing outliers ⇔ removing rows, and we don't want to remove rows from test set)
#outliners vanya jun cahi auru vanda different xan.. yo chai error in data collection or kunai external reason le garda hunxa.. kaile kai ko kura le effect parna vayena
#for example aauta ghar ma oil vetiyo vandaima sab ko value badni ta hoina


newTrain = data.iloc[:1460]
newTest = data.iloc[1460:]


#Second, we will define a function that returns outlier values using percentile() method

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75]) #Get 1st and 3rd quartiles (25% -> 75% of data will be kept)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5) #Get lower bound
    upper_bound = quartile_3 + (iqr * 1.5) #Get upper bound
    return np.where((ys > upper_bound) | (ys < lower_bound)) #Get outlier values

#Third, we will drop the outlier values from the train set

trainWithoutOutliers = newTrain #We can't change train while running through it

for column in newTrain:
    outlierValuesList = np.ndarray.tolist(outliers_iqr(newTrain[column])[0]) #outliers_iqr() returns an array
    trainWithoutOutliers = newTrain.drop(outlierValuesList) #Drop outlier rows
    
trainWithoutOutliers = newTrain


X = trainWithoutOutliers.drop("SalePrice", axis=1) #Remove SalePrice column
Y = np.log1p(trainWithoutOutliers["SalePrice"]) #Get SalePrice column {log1p(x) = log(x+1)}

x_test = X[:400]
y_test = Y[:400]

linear = linear_model.LinearRegression()
linear.fit(X,Y)
acc = linear.score(x_test,y_test)
print("Accuracy",acc)

newTest = newTest.drop("SalePrice", axis=1) #Remove SalePrice column

predictions = np.expm1(linear.predict(newTest))

output = pd.DataFrame({'Id': newTest.Id, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
