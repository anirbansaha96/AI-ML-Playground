################################################################################################
import pandas as pd
import os
os.chdir('C:\\Users\\anirb\\Downloads\\titanic')
train_data=pd.read_csv('train.csv',index_col=0)
test_data=pd.read_csv('test.csv',index_col=0)
train=train_data.copy()
test=test_data.copy()
################################################################################################

################################### Finding Null Values ###########################################
null_columns_train=train.columns[train.isnull().any()]
number_null_train=train[null_columns_train].isnull().sum()
null_columns_test=test.columns[test.isnull().any()]
number_null_test=test[null_columns_test].isnull().sum()
################################################################################################

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
train['Cabin'] = train.apply(lambda x: 0 if (x['Cabin']=='None') else 1, axis=1)
test['Cabin'] = test.apply(lambda x: 0 if (x['Cabin']=='None') else 1, axis=1)

del(null_columns_train,number_null_train,null_columns_test,number_null_test)
del(train_data,test_data)
################################################################################################
from sklearn.model_selection import train_test_split
train,valid=train_test_split(train,test_size=0.2)
################################################################################################
import category_encoders as ce
cat_features = ['Sex','Embarked']
target_enc = ce.TargetEncoder(cols=cat_features)
target_enc.fit(train[cat_features], train['Survived'])
train = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))
test = test.join(target_enc.transform(test[cat_features]).add_suffix('_target'))
del(cat_features)
#########################        FEATURE SCALING        ############################################
feature_cols = ['Pclass', 'Sex_target', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked_target']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train[feature_cols]=sc.fit_transform(train[feature_cols])
test[feature_cols]=sc.transform(test[feature_cols])
valid[feature_cols]=sc.transform(valid[feature_cols])
####################################################################################################

####################################################################################################
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(train[feature_cols],train['Survived'])
valid_pred=classifier.predict(valid[feature_cols])
from sklearn.metrics import f1_score
f1_score(valid['Survived'], valid_pred, average='weighted')
from sklearn.metrics import confusion_matrix
A=confusion_matrix(valid['Survived'], valid_pred)
correct_percentage=(A.trace())/(valid['Survived'].count())
print(correct_percentage)

####################################################################################################
####################################################################################################
index1=test.index
y_pred=classifier.predict(test[feature_cols])
output=pd.DataFrame(data=y_pred,
                    index=index1,
                    columns={'Survived'})
output.to_csv('file_submit8_target_encoding.csv') 
####################################################################################################