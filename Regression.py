from data_preprocessing import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def simple_linear_regression(file, start):
    X_train, X_test, y_train, y_test = data_preprocessing('Data Directory\simple_linear_regression.csv', False, False,
                                                          False, [], start=start)
    regressor = train_simple_linear_regression_model(X_train, y_train)
    y_pred = predict_slr_model(regressor, X_test)
    visualization(X_train, y_train, X_train, X_train, 'red', 'blue', 'Salary vs Expectation (Training Set)', regressor,
                  'Experience (Years)', 'Salary', True, 'Images/Simple_Training_Set.png')
    visualization(X_test, y_test, X_train, X_train, 'red', 'blue', 'Salary vs Expectation (Test Set)', regressor,
                  'Experience (Years)', 'Salary', True, 'Images/Simple_Test_Set.png')

def train_simple_linear_regression_model(X_train, y_train):
    regressor =  LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def predict_slr_model(regressor, X_test):
    return regressor.predict(X_test)

def visualization(var1, var2, var3, var4, color, color2, title, regressor, x_label, y_label, save, file):
    plt.scatter(var1, var2, color=color)
    plt.plot(var3, regressor.predict(var4), color=color2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig(file)
    plt.show()

def multi_linear_regression(file, field, start):
    X_train, X_test, y_train, y_test = data_preprocessing('Data Directory\multiple_linear_regression.csv', False, True,
                                                          False, field, start=start)
    regressor = train_simple_linear_regression_model(X_train, y_train)
    regressor.fit(X_train, y_train)
    y_pred = predict_slr_model(regressor, X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

def polynomial_regression(file, start):
    X, y = import_dataset(file, start)
    # lin_reg = LinearRegression()
    # lin_reg.fit(X, y)
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    visualization(X, y, X, poly_reg.fit_transform(X), 'red', 'blue', 'Truth or Bluff on Salary', lin_reg_2, 'Position', 'Salary', True,
                  'Data Directory/polynomial_regression.png')
    prediction = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
    print(prediction)

def linear_support_vector_regression(file, start):
    X, y = import_dataset(file, start)
    # FEATURE SCALING
    y = y.reshape(len(y), 1)    # Convert to 2D array
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    regressor = SVR(kernel='rbf')
    regressor.fit(X, y)
    # REVERSE THE SCALE AND PASS THE PREDICTION INTO y TO GET THE NEW RESULT WITH A RESHAPE OF DATA
    sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
    # VISUALIZE
    plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
    plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.savefig('Images/linear_SVR.png')
    plt.show()


def decision_tree_regression(file, start):
    X, y = import_dataset(file, start)
    # Training DTR on entire dataset
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    regressor.predict([[6.5]])
    visualization(X, y, X, X, 'red', 'blue', 'Truth or Bluff (Decision Tree Regression)', regressor, 'Position Level',
                  'Salary', True, 'Images/Decision_Tree_Regresssor.png')

def random_forest_regression(file, start):
    X, y = import_dataset(file, start)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X, y)
    regressor.predict([[6.5]])
    visualization(X, y, X, X, 'red', 'blue', 'Truth or Bluff (Random Forest Regression)', regressor, 'Position Level',
                  'Salary', True, 'Images/Random_Forest_Regressor.png')

def get_r2_score(y_test, y_pred):
    return r2_score(y_test, y_pred)