#!/bin/python
import sys
sys.path.append("../datamanger");
from dataloader import *
from itemsim_calculater import *
from numpy import *
from sklearn.metrics import mean_squared_error

class ItemBased_CF_Predictor:
    def __init__(self, ds, isc, topn,weight_type):
        self.ds = ds
        self.isc = isc
        self.topn = topn
        self.weight_type = weight_type
        self.ratings_guess = []
        self.ratings_test_value = []

    def init_ratings_test(self):
        for i in range(0,len(self.ds.ratings_test)):
            self.ratings_test_value.append(self.ds.ratings_test[i].rating)

    def predict(self):
        self.ratings_guess = []
        for i in range(0,len(self.ds.ratings_test)):
            pridicted_score = self.guess(self.ds.ratings_test[i].user_id - 1,self.ds.ratings_test[i].item_id - 1)
            self.ratings_guess.append(pridicted_score)
            
    def guess(self, userid, itemid):
        item_similarity = {}
        for i in range(0,self.ds.n_items):
            if not i == userid:
                item_similarity[i] = self.isc.item_sim[itemid][i]
        item_similarity_list =  sorted(item_similarity.items(), lambda x, y: cmp(y[1], x[1]))

        total_score = 0.0
        count = 0.0
        index = 0
        while count < self.topn and index < (self.ds.n_items - 1):
            if item_similarity_list[index][1] <= 0:
                break
            if self.ds.utility[userid][item_similarity_list[index][0]] > 0:
                if self.weight_type == 0:
                    count += 1
                    total_score += self.ds.utility_normal[userid][item_similarity_list[index][0]]
                elif self.weight_type == 1:
                    count += self.isc.item_sim[itemid][item_similarity_list[index][0]]
                    total_score += self.ds.utility_normal[userid][item_similarity_list[index][0]]*self.isc.item_sim[itemid][item_similarity_list[index][0]]
                
            index = index + 1
        if count <= 0:
            return self.ds.items[itemid].avg_r + self.ds.users[userid].avg_r - self.ds.global_mean

        return total_score/count + self.ds.items[itemid].avg_r + self.ds.users[userid].avg_r - self.ds.global_mean

    def mse(self):
        return mean_squared_error(self.ratings_guess, self.ratings_test_value)

    def process(self):
        self.init_ratings_test()
        self.predict()
        print "IBCF Method,ScoreWeight Type:",str(self.weight_type),"TopN:",self.topn,"Mean of Squear Error:",self.mse()

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
    isc = ItemSim_Calculater(ds,1)
    print "itemsim init sucess"
    
    ucf = ItemBased_CF_Predictor(ds,isc,50,0)
    ucf.process()

    ucf = ItemBased_CF_Predictor(ds,isc,50,1)
    ucf.process()

