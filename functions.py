from wrangle import *


# Visualize your success!
import seaborn as sns
import matplotlib.pyplot as plt

#Stats
from scipy import stats

# sklearn for modeling:
from sklearn.tree import DecisionTreeClassifier,\
export_text, \
plot_tree
from sklearn.metrics import accuracy_score, \
classification_report, \
confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression





df = prep_telco()


train, validate, test =  train_validate_test(df, 'churn_Yes')

x_cols = ['payment_type_Electronic check','internet_service_type_Fiber optic','senior_citizen','contract_type_Month-to-month','online_security_Yes', 'dependents_No', 'monthly_charges']

y_cols = 'churn_Yes'



x_train = train[x_cols]
y_train = train[y_cols]

x_val = validate[x_cols]
y_val = validate[y_cols]

x_test = test[x_cols]
y_test = test[y_cols]

baseline_accuracy = ((train.churn_Yes) == 0).mean()

def get_decisionTree_model():
    """
    Returns a decision treen model with a max depth of 5
    prints out the Accuracy of train and validate and the 
    classification report
    """
    clf = DecisionTreeClassifier(max_depth=5, random_state=706)
    #class_weight='balanced'
    # fit the thing
    clf.fit(x_train, y_train)

    model_proba = clf.predict_proba(x_train)
    model_preds = clf.predict(x_train)

    model_score = clf.score(x_train, y_train)

    #classification report:
    print(
        classification_report(y_train,
                          model_preds))
    print('Accuracy of Random Tree classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train)))
    print('Accuracy of Random Tree classifier on validation set: {:.2f}'
     .format(clf.score(x_val, y_val)))
    return clf


def get_random_forest():
    """
    Runs through two for loops from range 1 - 5 each time increasing the max depth 
    and min sample leaf
    puts all of the models in a pandas data frame and sorts for the hightes valadation 
    Prints out the classification report on the best model
    """
    
    model_list = []

    for j in range (1, 5):
        for i in range(2, 5):
            rf = RandomForestClassifier(max_depth=i, min_samples_leaf=j, random_state=706)

            rf = rf.fit(x_train, y_train)
            train_accuracy = rf.score(x_train, y_train)
            validate_accuracy = rf.score(x_val, y_val)
            model_preds = rf.predict(x_train)

            output = {
                "min_samples_per_leaf": j,
                "max_depth": i,
                "train_accuracy": train_accuracy,
                "validate_accuracy": validate_accuracy,
                'model_preds': model_preds
            }
            model_list.append(output)
            
    df = pd.DataFrame(model_list)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    df["baseline_accuracy"] = baseline_accuracy
    # df[df.validate_accuracy > df.baseline_accuracy + .05].sort_values(by=['difference'], ascending=True).head(15)
    df.sort_values(by=['validate_accuracy'], ascending=False).head(1)
    
    #classification report:
    print(classification_report(y_train, df['model_preds'][1]))
    return df.sort_values(by=['validate_accuracy'], ascending=False).head(1)


def get_logReg_model(data):
    """
    build a logistical regression model and prints out the accuracy on training and validation along with the classification report. 
    Must type in train_val as your data arrg to get the train val result.
    Type test if you want to test the model
    """
    logit = LogisticRegression()
    logit.fit(x_train, y_train)
    y_pred = logit.predict(x_train)
    y_proba = logit.predict_proba(x_train)
    logit_val = logit.predict(x_val)
    if data == 'train_val':
        print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
         .format(logit.score(x_train, y_train)))
        print('Accuracy of Logistic Regression classifier on validation set: {:.2f}'
         .format(logit.score(x_val, y_val)))
        print(
        classification_report(y_train,
                          y_pred))
    else: 
        print('Accuracy of logistic regression classifier on test set: {:.2f}'
     .format(logit.score(x_test, y_test)))
    
    
def get_churn_pie():
    '''
    prints a pie chart for telco churn
    '''
    labels = ["Not churn", "Churn"]
  
    sns.set(style="whitegrid")
    plt.figure(figsize=(6,6))
    plt.pie(train.churn_Yes.value_counts(), labels=labels, autopct='%1.1f%%')
    plt.title('Telco Churn ')
    plt.show()
    

def get_driver(driver):
    """
    builds a univariant visual based on the driver you put as the arrg
    prints it out
    """
    print(f'Univariate assessment of feature {driver}:')
    sns.countplot(data=train, x=driver)
    plt.show()
    print(
    pd.concat([train[driver].value_counts(),
    train[driver].value_counts(normalize=True)],
    axis=1))
        
        
def get_chisquared(driver):
    """
    Does the chisquaired testing and prints out a visual that goes along with it.
    prints either reject null hypothesis or not
    """
    a = 0.05
    sns.barplot(data=train,
                x=driver,
                y=train['churn_Yes'],
               ci=False)
    plt.xticks(rotation='vertical')
    plt.title(f'Amount of churn based on {driver}')
    plt.show()
    observed = pd.crosstab(train[driver], train['churn'])
    chi2, p, _, hypothetical = stats.chi2_contingency(observed)
    if p < a:
        print(f'We can reject our null hypothesis: and say that contract type can be a driver because the p value: {p} is less that alpha: {a}')
    else:
        print('We have failed to reject our null hypothesis')
        
        
        
def get_monthly_charges():
    """
    does the t test stats test for monthly charges and churn
    also builds a boxplot visual to go with it
    """
    a = 0.05
    sns.boxplot(x=train.monthly_charges, y=train.churn)
    plt.show()
    
    churn = train[train.churn == 'Yes']
    churn = churn.monthly_charges
    noChurn = train[train.churn == "No"]
    noChurn = noChurn.monthly_charges
    
    t_stat, p = stats.ttest_ind(churn,
                                noChurn,
                                equal_var=True)
    if p < a:
        print(f'We can reject our null hypothesis: and say that being a monthly charges can be a driver because the p value: {p} is less that alpha: {a}')
    else:
        print('We have failed to reject our null hypothesis')
        
        
        
        