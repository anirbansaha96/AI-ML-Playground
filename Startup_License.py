#pip install fuzzywuzzy
#pip install python-Levenshtein
import pandas as pd
import os
from fuzzywuzzy import fuzz 
import numpy as np
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.model_selection import train_test_split
#from fuzzywuzzy import process

################################################################################################
os.chdir('C:\\Users\\anirb\\OneDrive\\Documents\\Placement\\Companies\\ZS')
train=pd.read_csv('train_file.csv',parse_dates=['APPLICATION CREATED DATE', 'APPLICATION REQUIREMENTS COMPLETE',
       'PAYMENT DATE','LICENSE TERM START DATE',
       'LICENSE TERM EXPIRATION DATE','LICENSE APPROVED FOR ISSUANCE',
       'DATE ISSUED', 'LICENSE STATUS CHANGE DATE'])
train.columns = [c.replace(' ', '_') for c in train.columns]
################################################################################################

#########################  NEW FEATURE TO COMPARE NAMES  #######################################
train['LEGAL_NAME']=train['LEGAL_NAME'].fillna('Unknown Value')
train['DOING_BUSINESS_AS_NAME']=train['DOING_BUSINESS_AS_NAME'].fillna('Unknown Value')
train['name_similarity'] = train.apply(lambda x: fuzz.partial_ratio(x['LEGAL_NAME'], x['DOING_BUSINESS_AS_NAME']), axis=1)
################################################################################################

##################################   NEW FEATURE TO COMPARE STATES    ########################################
#train['STATE'].value_counts() shows that IL:80546 and IN:1749 and everything else is almost below 500 and 54 different states.
#So we create a STATE_Feature column with three categories IL,IN and everything else
train['State_Feature'] = train.apply(lambda x: x['STATE'] if (x['STATE'] in ['IL','IN']) else 'Everything Else', axis=1)
################################################################################################

#########################   NEW FEATURE TO COMPARE LICENSE CODES    ########################################
#train['LICENSE CODES'].value_counts() shows that 1010:50078 and 1011:10633 and everything else is almost below 3000 and 106 different states.
#So we create a LICENSE_CODE_Feature column with three categories 1010,1011 and everything else
train['LICENSE_CODE_Feature'] = train.apply(lambda x: x['LICENSE_CODE'] if (x['LICENSE_CODE'] in [1010,1011]) else 0, axis=1)
################################################################################################


#############################        CREATING FEATURES USING DATES           ###################
train['APPLICATION_CREATED_DATE'] = train.apply(lambda x: x['APPLICATION_REQUIREMENTS_COMPLETE'] if (x['APPLICATION_TYPE']=='RENEW') else x['APPLICATION_CREATED_DATE'], axis=1)
train = train.assign(duration1=(train.APPLICATION_REQUIREMENTS_COMPLETE-train.APPLICATION_CREATED_DATE).dt.days)
train = train.assign(duration2=(train.LICENSE_TERM_EXPIRATION_DATE-train.LICENSE_TERM_START_DATE).dt.days)
train = train.assign(duration3=(train.DATE_ISSUED-train.LICENSE_APPROVED_FOR_ISSUANCE).dt.days)
train = train.assign(duration4=(train.LICENSE_TERM_START_DATE-train.APPLICATION_CREATED_DATE).dt.days)
train = train.assign(duration5=(train.PAYMENT_DATE-train.APPLICATION_CREATED_DATE).dt.days)
train = train.assign(duration6=(train.PAYMENT_DATE-train.APPLICATION_REQUIREMENTS_COMPLETE).dt.days)
####################################################################################################

#null_columns=X.columns[X.isnull().any()]
#number_null=X[null_columns].isnull().sum()

####################################################################################################
for cols in ['duration1','duration2','duration3','duration4','duration5','duration6']:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp=imp.fit(train[[cols]])
    train[cols] = imp.transform(train[[cols]]).ravel()
####################################################################################################
    
####################################################################################################
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
train_application_type_ohe = ohe.fit_transform(train[['APPLICATION_TYPE','CONDITIONAL_APPROVAL','State_Feature','LICENSE_CODE_Feature']])
####################################################################################################
   

####################################################################################################
y=train['LICENSE_STATUS']
X=train[['duration1','duration2','duration3','duration4','duration5','duration6','name_similarity']]
X = pd.concat([X, train_application_type_ohe], axis=1)
####################################################################################################


##### FEATURE SCALING #####

from sklearn.preprocessing import StandardScaler
StandardScaler=StandardScaler()
X=StandardScaler.fit_transform(X)

###########################

#########################################################################
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')
#########################################################################


####################################################################################################
test=pd.read_csv('test_file.csv',parse_dates=['APPLICATION CREATED DATE', 'APPLICATION REQUIREMENTS COMPLETE',
       'PAYMENT DATE','LICENSE TERM START DATE',
       'LICENSE TERM EXPIRATION DATE','LICENSE APPROVED FOR ISSUANCE',
       'DATE ISSUED', 'LICENSE STATUS CHANGE DATE'])
test.columns = [c.replace(' ', '_') for c in test.columns]
test['LEGAL_NAME']=test['LEGAL_NAME'].fillna('Unknown Value')
test['DOING_BUSINESS_AS_NAME']=test['DOING_BUSINESS_AS_NAME'].fillna('Unknown Value')
test['name_similarity'] = test.apply(lambda x: fuzz.partial_ratio(x['LEGAL_NAME'], x['DOING_BUSINESS_AS_NAME']), axis=1)
test['State_Feature'] = train.apply(lambda x: x['STATE'] if (x['STATE'] in ['IL','IN']) else 'Everything Else', axis=1)
test['APPLICATION_CREATED_DATE'] = test.apply(lambda x: x['APPLICATION_REQUIREMENTS_COMPLETE'] if (x['APPLICATION_TYPE']=='RENEW') else x['APPLICATION_CREATED_DATE'], axis=1)
test['LICENSE_CODE_Feature'] = test.apply(lambda x: x['LICENSE_CODE'] if (x['LICENSE_CODE'] in [1010,1011]) else 0, axis=1)
test = test.assign(duration1=(test.APPLICATION_REQUIREMENTS_COMPLETE-test.APPLICATION_CREATED_DATE).dt.days)
test = test.assign(duration2=(test.LICENSE_TERM_EXPIRATION_DATE-test.LICENSE_TERM_START_DATE).dt.days)
test = test.assign(duration3=(test.DATE_ISSUED-test.LICENSE_APPROVED_FOR_ISSUANCE).dt.days)
test = test.assign(duration4=(test.LICENSE_TERM_START_DATE-test.APPLICATION_CREATED_DATE).dt.days)
test = test.assign(duration5=(test.PAYMENT_DATE-test.APPLICATION_CREATED_DATE).dt.days)
test = test.assign(duration6=(test.PAYMENT_DATE-test.APPLICATION_REQUIREMENTS_COMPLETE).dt.days)
for cols in ['duration1','duration2','duration3','duration4','duration5','duration6']:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp=imp.fit(test[[cols]])
    test[cols] = imp.transform(test[[cols]]).ravel()
#ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
test_application_type_ohe = ohe.transform(test[['APPLICATION_TYPE','CONDITIONAL_APPROVAL','State_Feature','LICENSE_CODE_Feature']])
X_test=test[['duration1','duration2','duration3','duration4','duration5','duration6','name_similarity']]
X_out = pd.concat([X_test, test_application_type_ohe], axis=1)
X_out=X_out.fillna(0)
X_out=StandardScaler.transform(X_out)
y_pred=classifier.predict(X_out)
output=pd.DataFrame(data=y_pred,
                    index=test['ID'],
                    columns={'LICENSE STATUS'})
output.to_csv('file_submit8.csv') 