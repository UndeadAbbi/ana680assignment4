import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv(xxx, header=None)

X = data.iloc[:, 2:] 
y = data.iloc[:, 1] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")


with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
