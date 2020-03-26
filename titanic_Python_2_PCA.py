################################################################################################
import pandas as pd
import os
import math
import seaborn as sns

os.chdir('C:\\Users\\anirb\\Downloads\\titanic')
train_data=pd.read_csv('train.csv',index_col=0)
test_data=pd.read_csv('test.csv',index_col=0)
train=train_data.copy()
test=test_data.copy()
################################################################################################

################################################################################################
#train.describe()
#test.describe()
################################################################################################

################################### Finding Null Values ###########################################
null_columns_train=train.columns[train.isnull().any()]
number_null_train=train[null_columns_train].isnull().sum()
null_columns_test=test.columns[test.isnull().any()]
number_null_test=test[null_columns_test].isnull().sum()
################################################################################################
# Checking for outliers
#sns.boxplot(x=train['Age'])
#sns.boxplot(x=train['SibSp'])
#sns.boxplot(x=train['Parch'])
#sns.boxplot(x=train['Fare'])

################################################################################################
# Now we see most of the Cabin Numbers is missing, this is because Cabin's are only alloted 
# to first class passengers, so for rest we fill it as None.
train['Cabin'] = train['Cabin'].fillna('None')
test['Cabin'] = test['Cabin'].fillna('None')

# Now we see two rows are missing the embarked row in Train Data we fill this using Modal values
train['Embarked']=train['Embarked'].fillna(train['Embarked'].mode().iloc[0])

# Now we see one row in Test data misses value of Fare, we fill it using mean value.
test['Fare']=test['Fare'].fillna(test['Fare'].mean())

# Now we seeembed missing values of Age, using mean value.
train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())
################################################################################################
# Now we take ceiling values for ages and fare to get lower number of classes of them
for i in range(0,len(train)):
    train['Age'].iat[i]=math.ceil(train['Age'].iat[i]);
    train['Fare'].iat[i]=math.ceil(train['Fare'].iat[i]);
for i in range(0,len(test)):
    test['Age'].iat[i]=math.ceil(test['Age'].iat[i]);
    test['Fare'].iat[i]=math.ceil(test['Fare'].iat[i]);

################################################################################################
import category_encoders as ce
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
train_category_ohe = ohe.fit_transform(train[['Sex','Embarked']])
test_category_ohe = ohe.transform(test[['Sex','Embarked']])
################################################################################################

# We differentiate between people with Cabins and those without cabins
train['Cabin'] = train.apply(lambda x: 0 if (x['Cabin']=='None') else 1, axis=1)
test['Cabin'] = test.apply(lambda x: 0 if (x['Cabin']=='None') else 1, axis=1)

####################################################################################################
y=train['Survived']
X=train[['Pclass','Age', 'SibSp', 'Parch','Fare','Cabin']]
X = pd.concat([X, train_category_ohe], axis=1)

X_out=test[['Pclass','Age', 'SibSp', 'Parch','Fare','Cabin']]
X_out = pd.concat([X_out, test_category_ohe], axis=1)
####################################################################################################

####################################################################################################
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
####################################################################################################


#########################        FEATURE SCALING        ############################################
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_out=sc.transform(X_out)
####################################################################################################

#########################        Applying PCA        ############################################
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
X_train=pca.fit_transform(X_train)
explained_variance_ratio=pca.explained_variance_ratio_
X_test=pca.transform(X_test)
X_out=pca.transform(X_out)
####################################################################################################

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')
from sklearn.metrics import confusion_matrix
A=confusion_matrix(y_test, y_pred)
correct_percentage=(A.trace())/(y_test.count())
print(correct_percentage)
#########################################################################

####################################################################################################
index1=test.index
y_pred=classifier.predict(X_out)
output=pd.DataFrame(data=y_pred,
                    index=index1,
                    columns={'Survived'})
output.to_csv('file_submit5_SVC.csv') 
####################################################################################################