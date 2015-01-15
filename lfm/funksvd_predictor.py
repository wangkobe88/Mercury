#!/bin/python
import sys
import random
sys.path.append("../datamanger");
from dataloader import *
from numpy import *
from sklearn.metrics import mean_squared_error

class FunkSVDBased_Predictor:
    def __init__(self, ds):
        self.ds = ds

        self.feature_dim = 120
        self.rated = zeros((self.ds.n_users, self.ds.n_items))
        self.predicted_score = zeros((self.ds.n_users, self.ds.n_items))
        self.features = zeros((self.ds.n_items,self.feature_dim))
        self.theta = zeros((self.ds.n_users, self.feature_dim))
        self.max_iterate_number = 50
        self.alpha = 0.0007
        self.lamda = 0.01

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
            for j in range(0,self.feature_dim):
                self.features[i][j] = random.random()/3

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
                    real_scores.append(self.ds.utility[i][j])

        #print predicted_scores
        #print real_scores
        return mean_squared_error(predicted_scores,real_scores)

    def guess(self, userid, itemid):
        score = 0.0
        for k in range(0,self.feature_dim):
            score += self.features[itemid][k]*self.theta[userid][k]
        return score

    def learning(self):
        pre_training_mse = 100.0
        for iter in range(0,self.max_iterate_number):
            training_mse = self.calculate_score()

            print "theta iterator:" + str(iter) + ",mse:",str(training_mse)
            if abs(pre_training_mse - training_mse ) < 0.0001:
                break
                
            pre_training_mse = training_mse
            for j in range(0, self.ds.n_users):
                for k in range(0,self.feature_dim):
                    descent = 0.0
                    for i in range(0, self.ds.n_items):
                        if not self.rated[j][i] == 0:
                            descent += (self.predicted_score[j][i] - self.ds.utility[j][i])* self.features[i][k]

                    self.theta[j][k] -= self.alpha * (descent + self.theta[j][k] * self.lamda)

            #training_mse = self.calculate_score()
            print "feature iterator:" + str(iter) + ",mse:",str(training_mse)
            
            for j in range(0, self.ds.n_items):
                for k in range(0,self.feature_dim):
                    descent = 0.0
                    for i in range(0, self.ds.n_users):
                        if not self.rated[i][j] == 0:
                            descent += (self.predicted_score[i][j] - self.ds.utility[i][j])* self.theta[i][k]

                    #if k == 0:
                    #    self.features[j][k] += self.alpha * descent
                    #else:
                    self.features[j][k] -= self.alpha * (descent + self.features[j][k] * self.lamda)
            
            self.alpha *= 0.96
            self.predict()
            print "Mean of Squear Error:",self.mse()

    def init_ratings_test(self):
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_test_value.append(self.ds.ratings_test[i].rating)

    def predict(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            itemid = self.ds.ratings_test[i].item_id - 1
            userid = self.ds.ratings_test[i].user_id - 1
            predicted_score = self.guess(userid,itemid)
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
    
    cbp = FunkSVDBased_Predictor(ds)
    cbp.process()

