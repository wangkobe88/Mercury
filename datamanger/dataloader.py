"""
Scripts to help load the movielens dataset into Python classes
@Author:www.wangke.me
@Date:20150103
"""
#!/bin/python
import re
from numpy import *

class User:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0

class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
    unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
    drama, fantasy, film_noir, horror, musical, mystery ,romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)
        self.avg_r = 0.0

class Rating:
    def __init__(self, user_id, item_id, rating, time):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.time = time

# The dataset class helps you to load files and create User, Item and Rating objects
class Dataset:
    def __init__(self,user_filename,item_filename,rating_filename,rating_test_filename):
        self.users = []
        self.items = []
        self.ratings = []
        self.ratings_test = []
        self.load_users(user_filename)
        self.load_items(item_filename)
        self.load_ratings(rating_filename)
        self.load_test_ratings(rating_test_filename)
        
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        
        self.utility = zeros((self.n_users, self.n_items))
        self.init_utility()

        self.global_mean = self.calculate_golobal_mean()
        self.calculate_items_mean()
        self.calculate_users_mean()

        self.utility_normal = zeros((self.n_users, self.n_items))
        self.init_utility_normal()

        self.utility_re_normal = zeros(( self.n_items , self.n_users))
        self.init_utility_re_normal()

    def init_utility_re_normal(self):
        for r in self.ratings:
            normal_score = r.rating + self.global_mean - self.items[r.item_id-1].avg_r - self.users[r.user_id-1].avg_r
            self.utility_re_normal[r.item_id-1][r.user_id-1] =  normal_score
    
    def init_utility_normal(self):
        for r in self.ratings:
            normal_score = r.rating + self.global_mean - self.items[r.item_id-1].avg_r - self.users[r.user_id-1].avg_r
            self.utility_normal[r.user_id-1][r.item_id-1] =  normal_score
        
    def init_utility(self):
        for r in self.ratings:
            self.utility[r.user_id-1][r.item_id-1] = r.rating

    def calculate_golobal_mean(self):
        totalscore = 0.0
        count = 0.0
        for i in range(0, self.n_items):
            for j in range(0,self.n_users):
                if self.utility[j][i] > 0:
                    totalscore += self.utility[j][i]
                    count += 1
        return totalscore/count
    
    def calculate_items_mean(self):
        for i in range(0, self.n_items):
            totalscore = 0.0
            count = 0.0
            for j in range(0, self.n_users):
                if self.utility[j][i] > 0:
                    totalscore += self.utility[j][i]
                    count += 1
            if totalscore > 0:
                self.items[i].avg_r = totalscore/count
            else:
                self.items[i].avg_r = self.global_mean

    def calculate_users_mean(self):
        for i in range(0, self.n_users):
            totalscore = 0.0
            count = 0.0
            for j in range(0,self.n_items):
                if self.utility[i][j] > 0:
                    count += 1
                    totalscore += self.utility[i][j]
            self.users[i].avg_r = totalscore/count

    def load_users(self,user_filename):
        user_file = open(user_filename, "r")
        text = user_file.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 5)
            if len(e) == 5:
                self.users.append(User(e[0], e[1], e[2], e[3], e[4]))
        user_file.close()

    def load_items(self, item_filename):
        item_file = open(item_filename, "r")
        text = item_file.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 24)
            if len(e) == 24:
                self.items.append(Item(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], \
                e[11], e[12], e[13], e[14], e[15], e[16], e[17], e[18], e[19], e[20], e[21], \
                e[22], e[23]))
        item_file.close()

    def load_ratings(self, rating_filename):
        rating_file = open(rating_filename, "r")
        text = rating_file.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                self.ratings.append(Rating(e[0], e[1], e[2], e[3]))
        rating_file.close()

    def load_test_ratings(self, rating_test_filename):
        rating_test_file = open(rating_test_filename, "r")
        text = rating_test_file.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                self.ratings_test.append(Rating(e[0], e[1], e[2], e[3]))
        rating_test_file.close()
        
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
