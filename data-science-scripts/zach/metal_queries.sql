/*
https://chartio.com/datarobot/explore/
*/

/*
BAD PROJECT NORM: 610193eebf7308c36dc6df2b
*/

/*
Given a PID, ID blueprints to try
*/
 /* TRY_TO_NUMBER(SAMPLE_LENGTH) <= 64 AND */
/* GSOD */
SET QUERY_PID = '610a877cb001c2e0e86bbe53';
SET METRIC = 'R Squared';
WITH project_blueprint_summary_raw AS (
  SELECT
    PROJECT_ID, BLUEPRINT_ID,
    MAX(ALL_METRICS_VALID_SCORE:"R Squared") AS R2
  FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD
  WHERE
    TRY_TO_NUMBER(ALL_METRICS_VALID_SCORE:"R Squared"::text) IS NOT NULL AND
    ALL_METRICS_VALID_SCORE:"R Squared" != 0 AND
    /*(hide_validation=0 or hide_validation is NULL) AND*/
    /*TRY_TO_NUMBER(SAMPLE_LENGTH) <= 64 AND*/
    PARTITION_INFO[0] != '(-1, -1)' AND
    IS_BLENDER = 0
  GROUP BY PROJECT_ID, BLUEPRINT_ID, IS_BLENDER
),
project_min_R2 AS (
  SELECT PROJECT_ID, min(R2) as min_R2
  FROM project_blueprint_summary_raw
  GROUP BY project_id
  ORDER BY min_R2
),
project_blueprint_summary AS (
  SELECT
    project_blueprint_summary_raw.project_id,
    project_blueprint_summary_raw.blueprint_id,
    project_blueprint_summary_raw.R2 AS R2_raw,
    CASE
      WHEN project_blueprint_summary_raw.R2 < 0 THEN -1 * project_blueprint_summary_raw.R2 / min_R2
      ELSE project_blueprint_summary_raw.R2 END AS R2
  FROM project_blueprint_summary_raw
  INNER JOIN project_min_R2 on project_blueprint_summary_raw.project_id = project_min_R2.project_id
),
project_norm AS (
  SELECT PROJECT_ID, SQRT(SUM(SQUARE(R2))) as NORM
  FROM project_blueprint_summary
  GROUP BY PROJECT_ID
  ORDER BY NORM DESC
),
query_project AS (SELECT * FROM project_blueprint_summary WHERE PROJECT_ID = $QUERY_PID),
similar_projects_raw AS (
  SELECT
    search_projects.PROJECT_ID,
    sum(query_project.R2 * search_projects.R2) AS SIMILARITY_RAW
  FROM query_project
  INNER JOIN project_blueprint_summary AS search_projects
    ON search_projects.BLUEPRINT_ID = query_project.BLUEPRINT_ID
    AND search_projects.PROJECT_ID != query_project.PROJECT_ID
  GROUP BY search_projects.PROJECT_ID
  ORDER BY SIMILARITY_RAW DESC
),
similar_projects AS (
  SELECT
    similar_projects_raw.PROJECT_ID,
    similar_projects_raw.SIMILARITY_RAW,
    search.NORM as search_norm,
    query.NORM as query_norm,
    search.NORM * query.NORM as norm_product,
    similar_projects_raw.SIMILARITY_RAW / (search.NORM * query.NORM) AS SIMILARITY
  FROM similar_projects_raw
  INNER JOIN project_norm as search on similar_projects_raw.PROJECT_ID = search.PROJECT_ID
  CROSS JOIN (SELECT NORM from project_norm WHERE PROJECT_ID = $QUERY_PID) as query
  ORDER BY SIMILARITY DESC
)
SELECT
  project_blueprint_summary.BLUEPRINT_ID,
  AVG(similar_projects.SIMILARITY * project_blueprint_summary.R2) AS score
FROM similar_projects
INNER JOIN project_blueprint_summary ON project_blueprint_summary.PROJECT_ID = similar_projects.PROJECT_ID
WHERE
	similar_projects.SIMILARITY > 0 AND
	project_blueprint_summary.BLUEPRINT_ID NOT IN (SELECT query_project.BLUEPRINT_ID FROM query_project)
GROUP BY project_blueprint_summary.BLUEPRINT_ID
ORDER BY score DESC

/*
ID MY PERSONAL PROJECTS that are in the DB
6100195295f81ac142697c37 - gsod
60ec99fa66b13f5a016456b6 - rec engine
61082ca175a6b64d4d9322ca - iso ne
6109ab00081a19878047d6a1 - Mike S shared with me
610a8785f5c69e1d976bbe4f - si ware lactose (comprehensive) - manually run a keras model here
610a877cb001c2e0e86bbe53 - si ware SNF (comprehensive) - DO NOT RUN MANUAL MODELS
6112bd1dd55ce3422bd5fa29 - mercari (soon)
6112bd68270d3074d8e7cdc6 - double pendulum (soon)

NOT IN DB:
# 60a6da70d1b6ae84cbdf86ce - common lit, text interpretation
# 60c247e7d52118e3719e0f29 - common lit with roberta embeddings addeds
# 5f2aed9fd6e13f04f6208eba - double pendulum
# 5eb965bf4414d7295713c70c - insurance dataset from json
# 5e3cd501f86f2d11bd249760 - basketball
# 5eb04af1bdede7087c81006b - mercari
*/

/* select all my projects */
select users.USERNAME, lb.CLUSTER_TYPE, lb.PROJECT_ID, prj.CREATION_TIME, prj.PROJECT_NAME, count(*) as NUMBER
from analytics.wall_e.STG_DR_APP_APP_LEADERBOARD as lb
inner join analytics.wall_e.STG_DR_APP_APP_PROJECTS as prj on lb.project_id = prj.id
inner join analytics.wall_e.STG_DR_APP_APP_USERS as users on prj.USER_ID = users.ID
where
	lb.BLUEPRINT_ID IS NOT NULL AND
	lb.ALL_METRICS_VALID_SCORE:"R Squared" IS NOT NULL AND
	users.USERNAME = 'zach@datarobot.com'
GROUP BY users.USERNAME, lb.CLUSTER_TYPE, lb.PROJECT_ID, prj.CREATION_TIME, prj.PROJECT_NAME
ORDER BY prj.CREATION_TIME DESC

/* map BP ID to json */
select BLUEPRINT from analytics.wall_e.STG_DR_APP_APP_LEADERBOARD
where BLUEPRINT_ID = 'a788003b12b0059dc331371773a8d377'
limit 10

/* Using predefined values to make a CTE*/
WITH cte AS (
  SELECT * FROM (
	VALUES
		('610a877cb001c2e0e86bbe53', 'ab8d9fb9bee89370a0e2148686faf086', 0.76355),
		('610a877cb001c2e0e86bbe53', 'c20ad236a47675c6eca4ece3e8ccfa01', 0.73494),
		('610a877cb001c2e0e86bbe53', '5493d14c7644ffd575a4e21edba839b8', 0.73494),
		('610a877cb001c2e0e86bbe53', 'eba99f9aec8a672dbb5976c2c48de37e', 0.72472),
		('610a877cb001c2e0e86bbe53', 'd5e1872d7ff7694e09ced2fd33c43044', 0.72349)

  ) t1 (PROJECT_ID, BLUEPRINT_ID, R2)
)
select * from cte

/*
Scratch pad
*/

select * from analytics.wall_e.STG_DR_APP_APP_PROJECTS as prj
INNER JOIN analytics.wall_e.STG_DR_APP_APP_LEADERBOARD as lb
ON prj.id = lb.project_id
where lb.BLUEPRINT is not NULL and lb.ALL_METRICS_VALID_SCORE is not NULL
limit 10

SELECT query_project.* FROM (
  SELECT lb.PROJECT_ID, lb.BLUEPRINT_ID, MAX(lb.ALL_METRICS_VALID_SCORE:"R Squared") AS R2
  FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD AS lb
  WHERE lb.PROJECT_ID = '610024cda9c178fc05a264ba'
  GROUP BY PROJECT_ID, BLUEPRINT_ID) as query_project

SELECT search_projects.* FROM
(SELECT lb.PROJECT_ID, lb.BLUEPRINT_ID, MAX(lb.ALL_METRICS_VALID_SCORE:"R Squared") AS R2
FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD AS lb
WHERE
	lb.ALL_METRICS_VALID_SCORE:"R Squared" IS NOT NULL AND
	TRY_TO_NUMBER(lb.SAMPLE_LENGTH) <= 64 AND
	lb.PROJECT_ID != '5eb04af1bdede7087c81006b' AND
	lb.BLUEPRINT_ID in
	  ('035a1fac44fe6f15f1928883ec0be2a7', '04f30e19ae845af6d38a0353ba4fe23d',
	  '0b106a4e9d4e274578a33ecbd578c9d4', '0bb5d4d3d84edab78dd006b7676922c6',
	  '0c19e571dc0b6ee448ac75e77496a4a7', '0e96a7576394dc532792d3ddda1ebe3d',
	  '0ee2bdc3f88cfc56b7d9785dd3d9d4bf', '0efe3da288f0a0ddc4aa4241805ca0c3',
	  '10fbb8b466b76a36a773e2e017f6aa4b', '11fd995bb6f7cf91f6491451d76e643b',
	  '15dcd1e98ce1d63ae48ef6a1dd6aa303', '191e2fe4cd76601dea86cab6d39c7323',
	  '1b30ef4abb03ed0e444214fed1fa9f17', '22a9298f51c1bd31ac0d110fd62e560b',
	  '22e3fd68512adf4d85d9b2bca0224c80', '23aa835b95f552472d941b3a2774f331',
	  '23cc61e00aabec08195a21775debc024', '268c9f22c2e992ff432c95db3166a611',
	  '28131ad4e05dc64a9e7a8d347a772c19', '2a3a1afee1dba173a8a03357e78abb74',
	  '2d2e6517f84b67a94964acd9317b05f8', '3424aa9e7137be8903d435095568eb0e',
	  '37a421b19d2a06e4bc1c03b28efb973f', '3b6f2b754c7b9188004462d52c4d1531',
	  '3b942c4a5fbf7ec77fc9d2fcb60a634e', '3f00d691680b556406d3db919bbcc3e3',
	  '41e82374e365659a51310d688b25373e', '46e4dd18bc96469d8703ee8a912df029',
	  '4a610f5d171afd7c91bd357271648bf8', '4bbba40a437ea5c48e48c5019bd8f06c',
	  '4cae68982af945686fbf15afde698d5f', '4d5f7c33a9160d73341c36096b06d8d2',
	  '4d6e072b80c0da82364cb5a8d9e3b83a', '53ba988af5948c4940dcfa8118b709d8',
	  '55ec4ac7352e3798559a33af796f5c83', '570ccee3730c29cef923d05d67b29fcc',
	  '5d79f4599e897297b5269ae3e617a7ad', '5e207293b7971f99bbca42cada12f90d',
	  '603b60bb57e0a0c3eb89ffa26c98c184', '6333c5dab73e5345dad1bec53034de5f',
	  '646a7a4aa9f2ce30134b9e7c38a90789', '658c9f7eae2111f97335531f700017bd',
	  '6681611f2f014fe54753da0d80b03ec1', '69e6faa1322f52cdb0e5cf79faf22994',
	  '6a69305a92ebadac8645e2a22ab16ef5', '6acb119e3da91bc1b5ceefedf2466b6b',
	  '6b6aa8b2c693f622db2a570c926f1d7f', '6b7f1ecdfd84d74f56531a6b02a2a04f',
	  '6c34d4f04eeef47ed9333217ba9ec2a8', '6c881873a647b90845363b8349ce6046',
	  '70c5e8d5ba4c3a9374cd519ba1a4fd72', '7105c6f4050aeec0cc193ca0346e6080',
	  '75b64613ff451a8cdfc81d840884e831', '76012462f51c6552e8d9ec6965a32cf1',
	  '76d0129e8e339bf31c7c2e444c4dbaa5', '77a65669e2188160d1194c09d2d9af84',
	  '77c1c5b93f4397ec780f9da562d7b497', '8108d7a4d6a1c9aefd75427fd3d50da9',
	  '814180275a55bcdb94e5f71b90b66bc2', '863d8cd9996dc670b3e962963d813ed2',
	  '8889fa4a1ee1f789ba0850acc307b14f', '8c2f27c2f9e952db45d0afa26cc0bf59',
	  '9107fa55d91e17a496fee392b78a2e68', '92e71f3120b9be83fe69404965edc96b',
	  '92fcc1ac4cad63d3b9d86d5fe616641d', '9328ba273809643d74397030eabdf455',
	  '93acc79a8a294abe64183d4a46c6b7f3', '94722e1c2518a8d22cfb21660643ade3',
	  '9de1c385aec4df9b722130cfde724030', 'a0f95fc840884784bb15a463e6232c3e',
	  'a17af9fa5c2e1a6a07693727019d2fb3', 'a21229336d7b8f2e422b567f68fe373f',
	  'a25ab1e71350333aef6a8e41177ef0f0', 'a30ac569a9a6ca187d077c4d00eb1ed3',
	  'aaaa500956e0e4fc88efa9ac35f6dc87', 'aad85c101f0c4973c6c86384b24dbf97',
	  'b0385b33f17886703f183dfe0194fdcd', 'b5472f672e82f5c784e3f1d1ec249098',
	  'b897b5f78b905b86aea140130a60250b', 'ba12cebba9e5740773848a48ea0c6be4',
	  'bb3473c2866689b47069020f8e0e7010', 'c06eb3a597e5dc9ae7e4e68d837aeaa5',
	  'c1ffc299a2e48e0ba7ec0591f5dc2f26', 'c20132fa537ff7e76ba93cd006614f0c',
	  'c951bab68ddb83de21180636d2799718', 'd3a6a5f4e863710fad1979aba10db5ce',
	  'd3b33a92556eb7b9f0532852ea7059e8', 'd42a211e1e26f7f0e3bf7b085e8171e0',
	  'db72cbd8c165d39905035ddf952257a7', 'db928410ddeb400a3cdf86344a67bd0b',
	  'dd73989eb167acb911da99f80eb09bf1', 'dd7a13b94fab6e9d26834b531ae782b3',
	  'e0af7e08092a17d0f8c30119c4e36e5c', 'e11dde87ee8fae8741d5920a23770e9e',
	  'e1703adac25d62c6ccf57b6d7ee5dfc2', 'e220b2c551983a67987e0b561b5bcd4f',
	  'e3b8fc13dae7c9254e7afd0c659a666c', 'e9eac2984bb6084b772e7a16e4e8cb86',
	  'f6be4aca923d76fd67d504d933aeeffc', 'f7bdfc401a29eca321e453225daec8e8',
	  'f9d9b2b0726d9496cefc7c3a7633fbf0', 'fc03ae7e096f1a01b209cc7bc14be40d')
GROUP BY PROJECT_ID, BLUEPRINT_ID
LIMIT 10) AS search_projects
