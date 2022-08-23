from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "person"
    
    id = Column("id", Integer, primary_key=True)
    username = Column("username", String, unique=True)    

# engine = create_engine("sqlite:///:memory:")
engine = create_engine("sqlite:///:memory:", echo=True)

# then do magics
Base.metadata.create_all(bind=engine)

# From the youtube vid tutorial commentator:
# quote
# session maker allows us to create a 
# session factory which is bound to
# the engine that we set up.
# unquote
Session = sessionmaker(bind=engine)

# ok, then instantiate it, was confused for a sec there
session = Session()

# let's add a User
user = User()
user.id = 0
user.username = "shooop"

# add to the session
session.add(user)
session.commit()

# query the users
users = session.query(User).all()
for user in users:
    print(user)
    print(user.username)

session.close()

