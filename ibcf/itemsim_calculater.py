"""
Scripts to help load the movielens dataset into Python classes
@Author:www.wangke.me
@Date:20150103
"""
#!/bin/python
import re
from numpy import *

class ItemSim_Calculater:
    def __init__(self,ds,sim_type):
        self.ds = ds
        self.sim_type = sim_type
        self.item_avg = zeros(ds.n_items)
        self.item_sim = zeros((ds.n_items, ds.n_items))
        self.init_item_avg()
        self.calculate_similarity()

    def init_item_avg(self):
        for i in range(0, self.ds.n_items):
            totalscore = 0.0
            count = 0.0
            for j in range(0,self.ds.n_users):
                totalscore += self.ds.utility_re_normal[i][j]
                count += 1
            self.item_avg[i] = totalscore/count

    def pcs(self, x, y):
        x_score = self.ds.utility_re_normal[x]
        y_score = self.ds.utility_re_normal[y]

        x_mean = self.item_avg[x]
        y_mean = self.item_avg[y]

        diffprod = 0
        xdiff2 = 0
        ydiff2 = 0
        for i in range(0,self.ds.n_users):
            if not (self.ds.utility[i][x] == 0 or self.ds.utility[i][y] == 0):
                xdiff = x_score[i] - x_mean
                ydiff = y_score[i] - y_mean
                diffprod += xdiff * ydiff
                xdiff2 += xdiff * xdiff
                ydiff2 += ydiff * ydiff
        if xdiff2 * ydiff2 == 0:
            return 0
        return diffprod / math.sqrt(xdiff2 * ydiff2)

    def cos(self,x, y):
        x_score = self.ds.utility_re_normal[x]
        y_score = self.ds.utility_re_normal[y]

        x_total_square = 0
        y_total_square = 0
        xy_cross_product =0

        for i in range(0,self.ds.n_users):
            if not (self.ds.utility[i][x] == 0 or self.ds.utility[i][y] == 0):
                xy_cross_product += x_score[i]*y_score[i]
                x_total_square +=  x_score[i]*x_score[i]
                y_total_square +=  y_score[i]*y_score[i]

        xy_sqrt_square_product = sqrt(x_total_square)*sqrt(y_total_square)
        if xy_sqrt_square_product == 0:
            return 0
        return xy_cross_product/xy_sqrt_square_product

    def euclidean(self,x, y):
        x_score = self.utility_re_normal[x]
        y_score = self.utility_re_normal[y]

        xy_minus_product = 0
        for i in range(0,self.ds.n_users):
            if not (self.ds.utility[i][x] == 0 or self.ds.utility[i][y] == 0) :
                xy_minus_product += (x_score[i]-y_score[i])*(x_score[i]-y_score[i])
        xy_sqrt_minus_product = sqrt(xy_minus_product)
        return xy_sqrt_minus_product

    def calculate_similarity(self):
        for i in range(0, self.ds.n_items):
            for j in range(0, self.ds.n_items):
                if self.item_sim[j][i] == 0:
                    if self.sim_type == 0:
                        self.item_sim[i][j] = self.pcs(i,j)
                    elif self.sim_type == 1:
                        self.item_sim[i][j] = self.cos(i,j)
                    elif self.sim_type == 2:
                        self.item_sim[i][j] = self.euclidean(i,j)
                else:
                    self.item_sim[i][j] = self.item_sim[j][i]

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
