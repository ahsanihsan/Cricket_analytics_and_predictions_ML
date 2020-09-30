import pandas as pd
import numpy as np
import yaml 
import csv
import sklearn
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime
from sklearn.impute import SimpleImputer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

directory = './datasets/PKvsIND_ODI/'
outcome_of_matches = []

for i,f in enumerate(os.listdir(directory)):
    data = yaml.safe_load(open(directory+f))['info']
    outcome_of_matches.append({
        'id': i+1,
        'city': data.get('city'),
        'month': ( datetime.strptime(data.get('dates')[0], "%Y-%m-%d") if type(data.get('dates')[0]) == str else data.get('dates')[0]).strftime("%B"),
        'match_type': data['match_type'],
        'team_a': data['teams'][0],
        'team_b': data['teams'][1],
        'toss_won': data['toss']['winner'],
        'toss_decision': data['toss']['decision'],
        'venue': data['venue'],
        'winner': data['outcome']['winner']
    })

f= open('./csv/PKvsIND_ODI.csv','w+')
w = csv.DictWriter(f, ['id','city','month','match_type','team_a','team_b','toss_won','toss_decision','venue','winner'])
w.writeheader()
for obj in outcome_of_matches:
    w.writerow(obj)
f.close()

data = pd.read_csv('./csv/PKvsIND_ODI.csv')
del data['id']
data.head()

encoder_dict = defaultdict(LabelEncoder)
label_data = data.apply(lambda x: encoder_dict[x.name].fit_transform(x))
label_data.head()

encoder_dict

label_data.iloc[[1,2]]

label_data.apply(lambda x: encoder_dict[x.name].inverse_transform(x)).head()

output = label_data.pop('winner')

x_train, x_test, y_train, y_test = train_test_split(label_data,output,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

dc_scores = []
dc_model = DecisionTreeClassifier()
folds = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in folds.split(label_data):
    xf_train, xf_test, yf_train, yf_test = label_data.iloc[train_index], label_data.iloc[test_index], output.iloc[train_index], output.iloc[test_index]
    dc_model.fit(xf_train, yf_train)
    dc_scores.append(dc_model.score(xf_test, yf_test))


print dc_scores

from sklearn.svm import SVC
svm_model = SVC()

svm_model.fit(x_train,y_train)

y_head_svm = svm_model.predict(x_test)

svm_score = svm_model.score(x_test,y_test)

svm_score

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)

y_head_lr = lr_model.predict(x_test)

lr_score = lr_model.score(x_test,y_test)

lr_score

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()

nb_model.fit(x_train,y_train)

y_head_nb = nb_model.predict(x_test)

nb_score = nb_model.score(x_test,y_test)

nb_score

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()

knn_model.fit(x_train,y_train)

y_head_knn = knn_model.predict(x_test)

knn_score = knn_model.score(x_test,y_test)

knn_score

nb_scores = []
nb_model = GaussianNB()
folds = KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in folds.split(label_data):
    xf_train, xf_test, yf_train, yf_test = label_data.iloc[train_index], label_data.iloc[test_index], output.iloc[train_index], output.iloc[test_index]
    nb_model.fit(xf_train, yf_train)
    nb_scores.append(nb_model.score(xf_test, yf_test))


print np.mean(nb_scores)

val = pd.DataFrame({
        'id':1,
        'city': 'Karachi',
        'month': 'january',
        'match_type': 'ODI',
        'team_a': 'Pakistan',
        'team_b': 'India',
        'toss_won': 'Pakistan',
        'toss_decision': 'bat',
        'venue': 'National Stadium',
    }, index=[0])
del val['id']
label_val = val.apply(lambda x: encoder_dict[x.name].fit_transform(x))
pred = nb_model.predict(label_val)
encoder_dict['winner'].inverse_transform(pred)

//////////////fr 2

file = open('./datasets/PKvsIND_ODI/_14.yml')

b = yaml.safe_load(file)

import copy
directory = './datasets/PKvsIND_ODI/'
played_balls = []
id=0
for i,f in enumerate(os.listdir(directory)):
    print(f)
    load = yaml.safe_load(open(directory+f))
    data= load['info']
    obj={
        'id': id,
        'city': data.get('city'),
        'month': ( datetime.strptime(data.get('dates')[0], "%Y-%m-%d") if type(data.get('dates')[0]) == str else data.get('dates')[0]).strftime("%B"),
        'match_type': data['match_type'],
        'venue': data['venue'],
        'batsman':'',
        'bowler':'',
        'over_no':'',
        'ball_no':'',
        'wicket_out':'',
        'score':'',
        'position':'',
        'team':''
        
    }
    for innings in load['innings']:
       
        first=innings[list(innings.keys())[0]]
        obj['team'] = first['team']
        first = first['deliveries'][0][0.1]
        p=3
        positions = {first['batsman']:1,first['non_striker']:2}
        for d in innings[list(innings.keys())[0]]['deliveries']:
            ball = str(list(d.keys())[0])
            over_ball=ball.split('.')
            obj['over_no']=over_ball[0]
            id+=1
            obj['wicket_out']=False
            obj['id']=id
            obj['ball_no']=over_ball[1]
            obj['bowler']=d[float(ball)]['bowler']
            obj['batsman']=d[float(ball)]['batsman']
            if obj['batsman'] in positions:
                obj['position']=positions[obj['batsman']]
            else:
                obj['position']=p
                positions[obj['batsman']]=p
                p+=1
                
            obj['score']=d[float(ball)]['runs']['total']
            if 'wicket' in d[float(ball)]:
                if obj['score']==0:
                    if obj['batsman']==d[float(ball)]['wicket']['player_out']:
                        obj['wicket_out']=True
                    else:
                        obj['wicket_out']=False
                        played_balls.append(obj)
                        obj['batsman']=d[float(ball)]['wicket']['player_out']
                        obj['score']=0

            else:
                d[float(ball)]['wicket_out'] = False
            played_balls.append(copy.deepcopy(obj))

f= open('./csv/PKvsIND_ODI_players.csv','w+')
w = csv.DictWriter(f, ['id','city','month','match_type','batsman','position','bowler','over_no','ball_no','venue','wicket_out','score'])
w.writeheader()
for obj in played_balls:
    w.writerow(obj)
f.close()

data = pd.read_csv('./csv/PKvsIND_ODI_players.csv').dropna()
data.head()

del data['id']

len(data)

data.head()

encoder_dict = defaultdict(LabelEncoder)
label_data = data.apply(lambda x: encoder_dict[x.name].fit_transform(x))
label_data.head()

X=label_data.iloc[:,0:10]
Y=label_data.iloc[:,10]
X.head()

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.svm import SVC
svm_model = SVC()

svm_model.fit(x_train,y_train)

y_head_svm = svm_model.predict(x_test)

svm_score = svm_model.score(x_test,y_test)

svm_score

nb_scores = []
nb_model = GaussianNB()
folds = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in folds.split(X):
    xf_train, xf_test, yf_train, yf_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    nb_model.fit(xf_train, yf_train)
    nb_scores.append(nb_model.score(xf_test, yf_test))


print np.mean(nb_scores)

input_d = pd.DataFrame({
        'city': 'Lahore',
        'month': 'February',
        'match_type': 'ODI',
        'team':'Pakistan',
        'batsman':'Sohail Khan',
        'position':1,
        'bowler':'S Sreesanth',
        'over_no':'0',
        'ball_no':'6',
        'venue': 'Gaddafi Stadium'
        
    },index=[0])
l_input_d=input_d.apply(lambda x: encoder_dict[x.name].fit_transform(x))
encoder_dict['wicket_out'].inverse_transform(nb_model.predict(l_input_d))