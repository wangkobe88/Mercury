"""
Predictor based on mean of item,mean of user and mean of global
@Author:www.wangke.me
@Date:20150103
"""

#!/bin/python
import sys
sys.path.append("../datamanger");
from dataloader import *
from numpy import *
from sklearn.metrics import mean_squared_error

class Normalized_Based_Predictor:
    def __init__(self, ds):
        self.ds = ds
        self.ratings_guess = []
        self.ratings_test_value = []

    def init_ratings_test(self):
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_test_value.append(self.ds.ratings_test[i].rating)

    def predict_based_meanofitems(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_guess.append(self.ds.items[self.ds.ratings_test[i].item_id - 1].avg_r)

    def predict_based_meanofusers(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_guess.append(self.ds.users[self.ds.ratings_test[i].user_id - 1].avg_r)

    def predict_based_meanofitemsandusers(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            pridicted_score = self.ds.users[self.ds.ratings_test[i].user_id - 1].avg_r + self.ds.items[self.ds.ratings_test[i].item_id - 1].avg_r - self.ds.global_mean
            self.ratings_guess.append(pridicted_score)

    def mse(self):
        return mean_squared_error(self.ratings_guess, self.ratings_test_value)

    def process(self):
        self.init_ratings_test()
        
        self.predict_based_meanofitems()
        print "ItemMean Method,Mean of Squear Error:",self.mse()

        self.predict_based_meanofusers()
        print "UserMean Method,Mean of Squear Error:",self.mse()

        self.predict_based_meanofitemsandusers()
        print "UserAndItemMean Method,Mean of Squear Error:",self.mse()

        
        
if __name__ ==  "__main__":

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

    nbp =  normalized_based_predictor(ds)
    nbp.process()
    
