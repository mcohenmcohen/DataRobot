/*
Template for Zach's meta learning blueprint reccomender.

Requires 4 input variables:

METRIC - which metric you want to maximize.  
E.g. "R Squared", "FVE Binomial", "FVE Poisson", "FVE Gamma", "FVE Tweedie", "Gini Norm", "AUC"
Should work with any metric where "larger is better".  Works better with "normalized metrics" that range 0-1.

QUERY_PID - the query PID we're looking for recommendations.
This PID will be excluded from the search to avoid self-recommendations.

PID_BID_METRIC_DATA - Pairs of values representing the leaderboard for the search project.  These
should take the form of (BLUEPRINT_ID, METRIC).  See metal_queries.R for an example of how to pull a
leaderboard from the public API and format PID_BID_METRIC_DATA.

QUERY_NORM - for the leaderboard pairs in PID_BID_METRIC_DATA, sqrt(sum(METRIC ^ 2)). Used to normalize
the "project similarity results"

This query is organized into a bunch of common table expressions (CTEs):

project_blueprint_summary_raw: A summary of the leaderboard for every project in the analytics database.
Has one row per PROJECT_ID, BLUEPRINT_ID pair and maps them to the metric.  Note that I currently take 
the MAX of the metric per pair, but there's probably a more sophisticated way to do this.

This CTE currently uses ALL_METRICS_VALID_SCORE.  It may be worth using ALL_METRICS_CV_SCORE if present, otherwise 
ALL_METRICS_VALID_SCORE.  I NEVER want to use ALL_METRICS_HOLDOUT_SCORE: I want holdout score to be useful to evaluate
the quality of the meta learning process.

Excluded data:
- rows where the metric of interest is NULL or 0.
- rows that were trained into the validation/holdout set, which is represented as (-1, -1).
- blenders (for now, I only want to reccomend single models)

project_min_R2: the minimum R2 per project.  Note that R2 can be negative, it's full range is (-Inf, +1). This is 
also true of all the "FVE" metrics, but not of AUC or GINI norm.

project_blueprint_summary: I want my metric for reccomendation to range from (-1, 1) so in this CTE I divide 
negative values by the min (per project) to make the negative side of the metric range from -1 to +1.
This logic can also probably be improved.

project_norm: The norm of each project.  Imagine a matrix where the rows are projects and the columns are blueprints.
The element of this matrix are the R2 values of each project/blueprint pair.  Most of the elements of this matrix are
0, and they range from -1 to +1.  This table computes the length of the "bluperint R2 vector" by project.  I will use 
it later to normalize project similarity.

query_project: This is wherte we insert PID_BID_METRIC_DATA, or pairs of (BLUEPRINT_ID, METRIC) that we're going
to use to search for reccomendations.

similar_projects_raw: This is the raw similarity.  We use the query BLUEPRINT_IDs to "search" the project_blueprint_summary
table. For each match, we multiple the R2 values.  So projects where the same blueprint IDs have the same R2 value will
have "high values" for the match.  We then sum the matches per project to compute raw similarity.

similar_projects: The problem with "similar_projects_raw" is that projects with more models will have higher similarity.
In this CTE, we normalize the similarity by the Norm of each project.  This gives us a cosine similarity between each
search project and the query project (in terms of blueprints and R2).  Projects with no blueprints in common will
have a cosine similarity of 0.  Projects with the exact same set of blueprints and the same R2 for each one will have
as cosine similaity of +1.  Projects where the models are "inverse" and the good models from one project are "bad" on
the other project will have a similarity of -1.  You can read more about cosine similarity on wikipedia:
https://en.wikipedia.org/wiki/Cosine_similarity

reccomended_blueprints: Finally, we grab every blueprint from each of the similar_projects, and rank them
by the average(similarity * R2).  This ranking metric should prefer blueprints that meet
2 criteria:

1. The project has a high cosine similarity
2. The blueprint has a high R2 on that similar project

Finally, with blueprint_map_with_duplicates/blueprint_map_with_duplicates_ranked/blueprint_map we generate a
mapping table that maps BLUEPRINT_ID to BLUEPRINT strings.  We use the longest string in the database (which
is the least-reduced version of the blueprint).  We depublicate versions of the BP with the same number of
characters by the count of times that BP has occured.

We merge blueprint_map to reccomended_blueprints to get the actualy blueprint JSON for the top 10 reccomended
blueprints.

So the BLUF here is:
1. Compute project similarity based on the blueprints in common and their R2 values
2. Reccomend blueprints that have high R2 values on highly similar projects

So far, I've found that this approach works really well in a narrow set of use cases.  Sometimes (many times)
it can't find a good blueprint, but it often can make reccomendations that score high on the leaderboard
(it just isn't always able to beat the autopilot).  These recs should still be interesting for making ensembles.

As we backfill for data into the analytics DB, I hope to be able to make good recs for more projects.

ROOM FOR IMPROVEMENTS:
- Better selection / summarization of blueprints for project_blueprint_summary_raw than MAX
- Figure out how to handle featurelists
- Figure out how to handle projects like 10k diabetes with lots of duplicate 
     * maybe dedupe by file name?
     * AVG(similar_projects.SIMILARITY * project_blueprint_summary.R2) should be somewhat robust to dupes?
     * Maybe drop projects from similar_projects where the similarity is exactly identical?
*/

WITH project_blueprint_summary AS (
  SELECT
    PROJECT_ID, BLUEPRINT_ID,
    ASINH(MAX(ALL_METRICS_VALID_SCORE:"{METRIC}")) AS R2
  FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD
  WHERE
    (
      (TRY_TO_NUMBER(ALL_METRICS_VALID_SCORE:"{METRIC}"::text) IS NOT NULL) OR 
      (TRY_TO_NUMBER(ALL_METRICS_VALID_SCORE:"Weighted {METRIC}"::text) IS NOT NULL) 
    ) AND
    (
     ALL_METRICS_VALID_SCORE:"{METRIC}" != 0 OR
     ALL_METRICS_VALID_SCORE:"Weighted {METRIC}" != 0 
    ) AND
    PARTITION_INFO[0] != '(-1, -1)' AND
    IS_BLENDER = 0 AND
    PROJECT_ID != '{QUERY_PID}'
  GROUP BY PROJECT_ID, BLUEPRINT_ID, IS_BLENDER
),
project_norm AS (
  SELECT PROJECT_ID, SQRT(SUM(SQUARE(R2))) as NORM
  FROM project_blueprint_summary
  GROUP BY PROJECT_ID
  ORDER BY NORM DESC
),
query_project AS (
  SELECT * FROM (
	  VALUES 
	    {PID_BID_METRIC_DATA}
  ) t1 (BLUEPRINT_ID, R2)
),
similar_projects_raw AS (
  SELECT
    search_projects.PROJECT_ID,
    sum(query_project.R2 * search_projects.R2) AS SIMILARITY_RAW
  FROM query_project
  INNER JOIN project_blueprint_summary AS search_projects
    ON search_projects.BLUEPRINT_ID = query_project.BLUEPRINT_ID
  GROUP BY search_projects.PROJECT_ID
  ORDER BY SIMILARITY_RAW DESC
),
similar_projects AS (
  SELECT
    similar_projects_raw.PROJECT_ID,
    similar_projects_raw.SIMILARITY_RAW,
    search.NORM as search_norm,
    similar_projects_raw.SIMILARITY_RAW / (search.NORM * {QUERY_NORM}) AS SIMILARITY
  FROM similar_projects_raw
  INNER JOIN project_norm as search on similar_projects_raw.PROJECT_ID = search.PROJECT_ID
  ORDER BY SIMILARITY DESC
),
reccomended_blueprints AS (
  SELECT
    project_blueprint_summary.BLUEPRINT_ID,
    AVG(similar_projects.SIMILARITY * project_blueprint_summary.R2) AS rec_mean,
    MEDIAN(project_blueprint_summary.R2) AS R2_med
  FROM similar_projects
  INNER JOIN project_blueprint_summary ON project_blueprint_summary.PROJECT_ID = similar_projects.PROJECT_ID
  WHERE
    similar_projects.SIMILARITY > 0.05 AND
    project_blueprint_summary.BLUEPRINT_ID NOT IN (SELECT query_project.BLUEPRINT_ID FROM query_project)
  GROUP BY project_blueprint_summary.BLUEPRINT_ID
  ORDER BY rec_mean DESC
),
blueprint_map_with_duplicates AS (
  SELECT BLUEPRINT_ID, BLUEPRINT, COUNT(*) AS N
  FROM analytics.wall_e.STG_DR_APP_APP_LEADERBOARD
  WHERE BLUEPRINT IS NOT NULL AND BLUEPRINT_ID IS NOT NULL
  GROUP BY BLUEPRINT_ID, BLUEPRINT
),
blueprint_map_with_duplicates_ranked AS (
  SELECT
    *, 
    LEN(BLUEPRINT) AS LEN,
    ROW_NUMBER() OVER(PARTITION BY BLUEPRINT_ID ORDER BY LEN(BLUEPRINT) DESC, N DESC) AS RANK
   FROM blueprint_map_with_duplicates
),
blueprint_map AS (
  SELECT BLUEPRINT_ID, BLUEPRINT
  FROM blueprint_map_with_duplicates_ranked
  WHERE RANK = 1
)
SELECT
  BLUEPRINT,
  /*.2966 + 0.8514*rec_mean + 0.1484*R2_med AS SCORE*/
  .2023 + 1.0364*rec_mean + 0.2704*GREATEST(R2_med, 0) AS SCORE,
  rec_mean,
  R2_med
FROM reccomended_blueprints
INNER JOIN BLUEPRINT_MAP 
	ON reccomended_blueprints.BLUEPRINT_ID = blueprint_map.BLUEPRINT_ID
ORDER BY SCORE DESC
LIMIT 10
