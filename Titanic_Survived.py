from sklearn.ensemble import RandomForestClassifier
import pandas as pd
train_data = pd.read_csv('train.csv')
y = train_data["Survived"]

test_data = pd.read_csv('test.csv')
features = ["Pclass", "Sex", "SibSp", "Parch"]
print(train_data[features])


#Can use this to get dummies function from pandas to change string data to numeric data
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

print(X)


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
x_t =X[:150]
y_t =y[:150]
acc = model.score(x_t,y_t)
print(f"Accuracy of :{acc}")
'''
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
'''
