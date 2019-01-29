from mongo.mongodb import mongodb

m = mongodb()
m.insertone()
x = m.getall()

print(x)
