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
    print(played_balls)
f= open('./csv/PKvsIND_ODI_players.csv','w+')
w = csv.DictWriter(f, ['id','city','month','match_type','team','batsman','position','bowler','over_no','ball_no','venue','wicket_out','score'])
w.writeheader()
for obj in played_balls:
    w.writerow(obj)
f.close()