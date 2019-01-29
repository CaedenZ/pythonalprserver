import pymongo

class mongodb:
    def __init__(self):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        print("get")
        self.mydb = myclient["mydatabase"]

    def insertone(self,collection = "test",data = {"test": "aaa"}):
        mycol = self.mydb[collection]
        x = mycol.insert_one(data)
        print("inserted")
        # return x.inserted_id

    def getall(self,collection = "test", query = {}):
        mycol = self.mydb[collection]
        x = mycol.find(query)
        print("get")
        return x

    def getone(self,collection = "test", query = {}):
        mycol = self.mydb[collection]
        x = mycol.find_one(query)

        return x
