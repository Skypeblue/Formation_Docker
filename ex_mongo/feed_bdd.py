from pymongo import MongoClient


client = MongoClient(
    host='0.0.0.0',
    port=27017
)

db = client["star-wars"]

person = [{"name": 'NUTE GUNRAY', "value": 24, "color": '#808080'},\
	{"name": 'PK-4', "value": 3, "color": '#808080'},\
	{"name": 'TC-14', "value": 4, "color": '#808080'},\
	{"name": 'OBI-WAN', "value": 147, "color": '#48D1CC'},\
	{"name": 'DOFINE', "value": 3, "color": '#808080'},\
	{"name": 'RUNE', "value": 10, "color": '#808080'},\
	{"name": 'TEY HOW', "value": 4, "color": '#808080'}]

new_person_2 = [{"name": 'Frédéric', "value": "Jean", "color": '#191970', "role": 'Machine Learning Engineer'}]

def add_in_mongo(users, col):
	for n in users:
		x = col.insert_one(n)
		print(str(n) + "->" + str(x.inserted_id))

#add_in_mongo(person,db["person"])
add_in_mongo(new_person_2, db["person"])
