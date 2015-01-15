#!/bin/python
import sys
import random
sys.path.append("../datamanger");
from dataloader import *
from numpy import *
from sklearn.metrics import mean_squared_error

class ContentBased_Predictor:
    def __init__(self, ds):
        self.ds = ds
        self.feature_dim = 20
        
        self.rated = zeros((self.ds.n_users, self.ds.n_items))
        self.predicted_score = zeros((self.ds.n_users, self.ds.n_items))
        self.features = zeros((self.ds.n_items,self.feature_dim))
        self.theta = zeros((self.ds.n_users, self.feature_dim))
        self.max_iterate_number = 20
        self.alpha = 0.003
        self.lamda = 100

        self.ratings_guess = []
        self.ratings_test_value = []

        self.init_rated()
        self.init_features()
        self.init_theta()
        self.init_ratings_test()

    def init_rated(self):
        for r in self.ds.ratings:
            self.rated[r.user_id-1][r.item_id-1] = 1
        print "rate count:",sum(self.rated)
            #print r.user_id-1,r.item_id-1,self.rated[r.user_id-1][r.item_id-1]

    def init_features(self):
        for i in range(0, self.ds.n_items):
            self.features[i][0] = 1
            self.features[i][1] = self.ds.items[i].unknown
            self.features[i][2] = self.ds.items[i].action
            self.features[i][3] = self.ds.items[i].adventure
            self.features[i][4] = self.ds.items[i].animation
            self.features[i][5] = self.ds.items[i].childrens
            self.features[i][6] = self.ds.items[i].comedy
            self.features[i][7] = self.ds.items[i].crime
            self.features[i][8] = self.ds.items[i].documentary
            self.features[i][9] = self.ds.items[i].drama
            self.features[i][10] = self.ds.items[i].fantasy
            self.features[i][11] = self.ds.items[i].film_noir
            self.features[i][12] = self.ds.items[i].horror
            self.features[i][13] = self.ds.items[i].musical
            self.features[i][14] = self.ds.items[i].mystery
            self.features[i][15] = self.ds.items[i].romance
            self.features[i][16] = self.ds.items[i].sci_fi
            self.features[i][17] = self.ds.items[i].thriller
            self.features[i][18] = self.ds.items[i].war
            self.features[i][19] = self.ds.items[i].western

    def init_theta(self):
        for i in range(0,self.ds.n_users):
            for k in range(0,self.feature_dim):
                self.theta[i][k] = random.random()/3
        
    def calculate_score(self):
        predicted_scores = []
        real_scores = []
        
        for i in range(0,self.ds.n_users):
            for j in range(0,self.ds.n_items):
                #print i,j,self.rated[i][j]
                if self.rated[i][j] > 0.0:
                    score = self.guess(i, j)
                    self.predicted_score[i][j] = score
                    predicted_scores.append(score)
                    real_scores.append(self.ds.utility_normal[i][j])

        #print predicted_scores
        #print real_scores
        return mean_squared_error(predicted_scores,real_scores)

    def guess(self, userid, itemid):
        score = 0.0
        for k in range(0,self.feature_dim):
            score += self.features[itemid][k]*self.theta[userid][k]
        return score

    def learning(self):
        for iter in range(0,self.max_iterate_number):
            training_mse = self.calculate_score()
            print training_mse
            if training_mse < 0.7:
                break
            
            for j in range(0, self.ds.n_users):
                for k in range(0,self.feature_dim):
                    descent = 0.0
                    for i in range(0, self.ds.n_items):
                        if not self.rated[j][i] == 0:
                            descent += (self.predicted_score[j][i] - self.ds.utility_normal[j][i])* self.features[i][k]

                    if k == 0:
                        self.theta[j][k] -= self.alpha * descent
                    else:
                        self.theta[j][k] -= self.alpha * (descent + self.theta[j][k] * self.lamda)

            self.alpha *= 0.9

    def init_ratings_test(self):
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_test_value.append(self.ds.ratings_test[i].rating)

    def predict(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            itemid = self.ds.ratings_test[i].item_id - 1
            userid = self.ds.ratings_test[i].user_id - 1
            predicted_score = self.ds.items[itemid].avg_r + self.ds.users[userid].avg_r - self.ds.global_mean + self.guess(userid,itemid)
            self.ratings_guess.append(predicted_score)

    def mse(self):
        return mean_squared_error(self.ratings_guess, self.ratings_test_value)

    def process(self):
        self.learning()
        self.predict()
        print "CB Method,ScoreWeight Type:","Mean of Squear Error:",self.mse()

if __name__ ==  "__main__":
    import sys
    #user_filename = sys.argv[1]
    #item_filename = sys.argv[2]
    #rating_filename = sys.argv[3]
    #rating_test_filename = sys.argv[4]

    user_filename = "../data/u.user"
    item_filename = "../data/u.item"
    rating_filename = "../data/u.base"
    rating_test_filename = "../data/u.test"

    ds = Dataset(user_filename,item_filename,rating_filename,rating_test_filename)
    print "global_mean: ",ds.global_mean
    
    cbp = ContentBased_Predictor(ds)
    cbp.process()

