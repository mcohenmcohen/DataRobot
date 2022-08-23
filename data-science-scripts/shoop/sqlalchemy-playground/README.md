# Basics of SQLAlchemy

Helpful video ([LINK](https://www.youtube.com/watch?v=OT5qJBINiJY))

- Look at example1.py and example2.py
- When you run example2.py, you get the following output:
```
2021-08-16 00:31:31,784 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2021-08-16 00:31:31,784 INFO sqlalchemy.engine.Engine PRAGMA main.table_info("person")
2021-08-16 00:31:31,784 INFO sqlalchemy.engine.Engine [raw sql] ()
2021-08-16 00:31:31,784 INFO sqlalchemy.engine.Engine PRAGMA temp.table_info("person")
2021-08-16 00:31:31,785 INFO sqlalchemy.engine.Engine [raw sql] ()
2021-08-16 00:31:31,785 INFO sqlalchemy.engine.Engine
CREATE TABLE person (
	id INTEGER NOT NULL,
	username VARCHAR,
	PRIMARY KEY (id),
	UNIQUE (username)
)


2021-08-16 00:31:31,785 INFO sqlalchemy.engine.Engine [no key 0.00010s] ()
2021-08-16 00:31:31,786 INFO sqlalchemy.engine.Engine COMMIT
2021-08-16 00:31:31,787 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2021-08-16 00:31:31,788 INFO sqlalchemy.engine.Engine INSERT INTO person (id, username) VALUES (?, ?)
2021-08-16 00:31:31,788 INFO sqlalchemy.engine.Engine [generated in 0.00017s] (0, 'shooop')
2021-08-16 00:31:31,789 INFO sqlalchemy.engine.Engine COMMIT
2021-08-16 00:31:31,789 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2021-08-16 00:31:31,791 INFO sqlalchemy.engine.Engine SELECT person.id AS person_id, person.username AS person_username
FROM person
2021-08-16 00:31:31,791 INFO sqlalchemy.engine.Engine [generated in 0.00011s] ()
<__main__.User object at 0x1048e2be0>
shooop
2021-08-16 00:31:31,791 INFO sqlalchemy.engine.Engine ROLLBACK
```
