rs.secondaryOk()
db.leaderboard.find(
    {"_id": {"$gt": ObjectId("5a49c0500000000000000000"), "$lt": ObjectId("606545400000000000000000")}, 'test.R Squared': {'$exists': true}}, 
    {_id: 0, "blueprint_id":1, "blueprint":1})