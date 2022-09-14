from pymongo import MongoClient


client = MongoClient(
    host='localhost',
    port=27017,
)

db = client["star-wars"]

def print_user(name):
	print("--------------"+name+"--------------")
	col = db[name]
	for x in col.find():
		print(x)
	print("\n")

print_user("person")