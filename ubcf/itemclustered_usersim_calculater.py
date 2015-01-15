"""
Scripts to help load the movielens dataset into Python classes
@Author:www.wangke.me
@Date:20150103
"""
#!/bin/python
import sys
sys.path.append("../datamanger");
from dataloader import *

import re
from numpy import *
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

class ItemClustered_UserSim_Calculater:
    def __init__(self,ds,sim_type,cluster_type,cluster_num):
        self.ds = ds

        self.cluster_type = cluster_type
        self.cluster_num = cluster_num
        self.user_cluster_totalscore = zeros((self.ds.n_users, cluster_num))
        self.user_cluster_count = zeros((self.ds.n_users, cluster_num))
        self.user_cluster_mean = zeros((self.ds.n_users, cluster_num))

        self.item_cluster_labels = zeros(self.ds.n_items)
        self.item_multi_cluster_probs = zeros((self.ds.n_items,cluster_num))
        self.user_cluster_average = zeros(self.ds.n_users)
        
        self.items_cluster()
        self.get_user_cluster_score()
        self.get_user_cluster_average()
        
        self.sim_type = sim_type        
        self.user_sim = zeros((self.ds.n_users, self.ds.n_users))
        self.calculate_similarity()

    def get_user_cluster_average(self):
        for i in range(0, self.ds.n_users):
            total = 0.0
            count = 0.0
            for j in range(0,self.cluster_num):
                total += self.user_cluster_mean[i][j]
                count += 1
            self.user_cluster_average[i] = total/count
            
    def items_cluster(self):
        if self.cluster_type == 0:
            self.items_kmeans_cluster()
        elif self.cluster_type == 1:
            self.items_em_cluster()
        elif self.cluster_type == 2:
            self.items_em_cluster()

    def items_em_cluster(self):
        gmm = GMM(n_components = self.cluster_num,
                    covariance_type='full', init_params='wc', n_iter = self.cluster_num*5)
        gmm.fit(self.ds.utility_re_normal)
        self.item_cluster_labels = gmm.predict(self.ds.utility_re_normal)
        self.item_multi_cluster_probs = gmm.predict_proba(self.ds.utility_re_normal)

    def items_kmeans_cluster(self):
        k_means = KMeans(init='k-means++', n_clusters = self.cluster_num, n_init = self.cluster_num*5)
        k_means.fit(self.ds.utility_re_normal)
        self.item_cluster_labels = k_means.labels_
        
    def get_user_cluster_score(self):
        for i in range(0, self.ds.n_users):
            for j in range(0, self.ds.n_items):
                if self.ds.utility[i][j] > 0:
                    if self.cluster_type == 0 or self.cluster_type == 1:
                        self.user_cluster_totalscore[i][self.item_cluster_labels[j]] += self.ds.utility_normal[i][j]
                        self.user_cluster_count[i][self.item_cluster_labels[j]] += 1.0
                    elif self.cluster_type == 2:
                         for k in range(0,self.cluster_num):
                             self.user_cluster_totalscore[i][k] += self.ds.utility_normal[i][j] * self.item_multi_cluster_probs[j][k]
                             self.user_cluster_count[i][k] += self.item_multi_cluster_probs[j][k]

        if self.cluster_type == 0 or self.cluster_type == 1:
            for i in range(0, self.ds.n_users):
                for j in range(0,self.ds.n_items):
                    if self.user_cluster_count[i][self.item_cluster_labels[j]] > 0:
                        self.user_cluster_mean[i][self.item_cluster_labels[j]] = self.user_cluster_totalscore[i][self.item_cluster_labels[j]]/self.user_cluster_count[i][self.item_cluster_labels[j]]
                    else:
                        self.user_cluster_mean[i][self.item_cluster_labels[j]] = 0
        elif self.cluster_type == 2:
            for i in range(0, self.ds.n_users):
                for k in  range(0, self.cluster_num):
                    if self.user_cluster_count[i][k] > 0:
                        self.user_cluster_mean[i][k] = self.user_cluster_totalscore[i][k]/self.user_cluster_count[i][k]
                    else:
                        self.user_cluster_mean[i][k] = 0
        
    def pcs(self, x, y):
        x_score = self.user_cluster_mean[x]
        y_score = self.user_cluster_mean[y]

        x_mean = self.user_cluster_average[x]
        y_mean = self.user_cluster_average[y]

        diffprod = 0
        xdiff2 = 0
        ydiff2 = 0
        for i in range(0,self.cluster_num):
            if not (x_score[i] == 0 or y_score[i] == 0):
                xdiff = x_score[i] - x_mean
                ydiff = y_score[i] - y_mean
                diffprod += xdiff * ydiff
                xdiff2 += xdiff * xdiff
                ydiff2 += ydiff * ydiff
        if xdiff2 * ydiff2 == 0:
            return 0
        return diffprod / math.sqrt(xdiff2 * ydiff2)

    def cos(self,x, y):
        x_score = self.user_cluster_mean[x]
        y_score = self.user_cluster_mean[y]

        x_total_square = 0
        y_total_square = 0
        xy_cross_product =0

        for i in range(0,self.cluster_num):
            if not (x_score[i] == 0 or y_score[i] == 0):
                xy_cross_product += x_score[i]*y_score[i]
                x_total_square +=  x_score[i]*x_score[i]
                y_total_square +=  y_score[i]*y_score[i]

        xy_sqrt_square_product = sqrt(x_total_square)*sqrt(y_total_square)
        if xy_sqrt_square_product == 0:
            return 0
        return xy_cross_product/xy_sqrt_square_product

    def euclidean(self,x, y):
        x_score = self.user_cluster_mean[x]
        y_score = self.user_cluster_mean[y]

        xy_minus_product = 0
        for i in range(0,self.cluster_num):
            if not (x_score[i] == 0 or y_score[i] == 0) :
                xy_minus_product += (x_score[i]-y_score[i])*(x_score[i]-y_score[i])
        xy_sqrt_minus_product = sqrt(xy_minus_product)
        return xy_sqrt_minus_product

    def calculate_similarity(self):
        for i in range(0, self.ds.n_users):
            for j in range(0, self.ds.n_users):
                if self.user_sim[j][i] == 0:
                    if self.sim_type == 0:
                        self.user_sim[i][j] = self.pcs(i,j)
                    elif self.sim_type == 1:
                        self.user_sim[i][j] = self.cos(i,j)
                    elif self.sim_type == 2:
                        self.user_sim[i][j] = self.euclidean(i,j)
                else:
                    self.user_sim[i][j] = self.user_sim[j][i]

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
