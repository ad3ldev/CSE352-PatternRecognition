from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd

df = pd.DataFrame(
    {
        'x1': [1, 2, 2, 2, 3, 3, 4, 4, 4, 5],
        'x2': [5, 6, 10, 12, 17, 12, 6, 5, 7, 10],
        'y': [10, 40, 50, 60, 70, 50, 30, 20, 40, 70]
    }
)

# Q1, b, i
model = LinearRegression()
X, y = df[['x1', 'x2']], df.y
model.fit(X, y)
print(model.intercept_, model.coef_, model.score(X, y), "\n")

# Q1, b, ii
for i in [0.1, 1, 10, 100]:
    clf = Ridge(alpha=i)
    clf.fit(X, y)
    print("Alpha: ", i)
    print(clf.intercept_, clf.coef_, clf.score(X, y), "\n")
