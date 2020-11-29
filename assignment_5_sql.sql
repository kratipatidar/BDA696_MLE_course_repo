CREATE TABLE IF NOT EXISTS bc_join_g AS
SELECT g.game_id, bc.batter, bc.Hit, bc.atBat, g.local_date 
FROM batter_counts bc 
JOIN game          g
ON   bc.game_id = g.game_id ;

CREATE TABLE IF NOT EXISTS rolling_ba_100_days AS
SELECT a.game_id, a.batter, SUM(IFNULL(b.Hit,0))/NULLIF(SUM(IFNULL(b.atBat,0)),0) AS BA, COUNT(*) AS cnt, a.local_date
FROM   bc_join_g a
JOIN   bc_join_g b
ON a.batter = b.batter
AND b.local_date > DATE_SUB(a.local_date, INTERVAL 100 DAY) AND a.local_date > b.local_date
WHERE a.game_id = 12560      -- comment out this line to get the rolling batting average for all the players across all games
GROUP BY a.game_id, a.batter, a.local_date;
                                                            
-- saving results to a csv file
                                                            
SELECT * FROM rolling_ba_100_days
INTO OUTFILE './results.csv' ;  
