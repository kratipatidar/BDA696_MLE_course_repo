
USE baseball;

# aggregating metrics

CREATE TABLE IF NOT EXISTS rolling_200_day AS
SELECT tbc1.team_id, tbc1.game_id, COUNT(*) AS cnt,
       SUM(tbc2.plateApperance) AS plateAppearance,
       SUM(tbc2.atBat) AS atBat,
       SUM(tbc2.Hit) AS Hit,
       SUM(tbc2.toBase) AS Base,
       SUM(tbc2.Batter_Interference) AS Batter_Interference,
       SUM(tbc2.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out,
       SUM(tbc2.Bunt_Pop_Out) AS Bunt_Pop_Out,
       SUM(tbc2.Catcher_Interference) AS Catcher_Interference,
       SUM(tbc2.`Double`) AS `Double`,
       SUM(tbc2.Double_Play) AS Double_Play,
       SUM(tbc2.Fan_interference) AS Fan_interference,
       SUM(tbc2.Field_Error) AS Field_Error,
       SUM(tbc2.Fielders_Choice) AS Fielders_Choice,
       SUM(tbc2.Fielders_Choice_Out) AS Fielders_Choice_Out,
       SUM(tbc2.Fly_Out) +  SUM(tbc2.Flyout) AS Fly_Out,
       SUM(tbc2.Force_Out) +  SUM(tbc2.Forceout) AS Force_Out,
       SUM(tbc2.Ground_Out) +  SUM(tbc2.Groundout) AS Ground_Out,
       SUM(tbc2.Grounded_Into_DP) AS Grounded_Into_DP,
       SUM(tbc2.Hit_By_Pitch) AS Hit_By_Pitch,
       SUM(tbc2.Home_Run) AS Home_Run,
       SUM(tbc2.Intent_Walk) AS Intent_Walk,
       SUM(tbc2.Line_Out) +  SUM(tbc2.Lineout) AS Line_Out,
       SUM(tbc2.Pop_Out) AS Pop_Out,
       SUM(tbc2.Runner_Out) AS Runner_Out,
       SUM(tbc2.Sac_Bunt) AS Sac_Bunt,
       SUM(tbc2.Sac_Fly) AS Sac_Fly,
       SUM(tbc2.Sac_Fly_DP) AS Sac_Fly_DP,
       SUM(tbc2.Sacrifice_Bunt_DP) AS Sacrifice_Bunt_DP,
       SUM(tbc2.Strikeout) AS Strikeout,
       SUM(tbc2.`Strikeout_-_DP`) AS `Strikeout_DP`,
       SUM(tbc2.`Strikeout_-_TP`) AS `Strikeout_TP`,
       SUM(tbc2.Single) AS Single,
       SUM(tbc2.Triple) AS Triple,
       SUM(tbc2.Triple_Play) AS Triple_Play,
       SUM(tbc2.Walk) AS Walk
FROM team_batting_counts tbc1
JOIN game g1 
ON   g1.game_id = tbc1.game_id 
JOIN team_batting_counts tbc2
ON   tbc1.team_id = tbc2.team_id 
JOIN game  g2
ON g2.game_id = tbc2.game_id
AND g2.type IN ('R')
AND g2.local_date < g1.local_date 
AND g2.local_date >= DATE_ADD(g1.local_date, INTERVAL -200 DAY)
GROUP BY tbc1.team_id, tbc1.game_id
ORDER BY tbc1.team_id ;

CREATE UNIQUE INDEX team_game ON rolling_200_day(team_id, game_id);


# coming up with new features

CREATE TABLE IF NOT EXISTS baseball_features AS
SELECT g.game_id, 
       g.home_team_id AS home_team_id,
       g.away_team_id AS away_team_id,
       r2dh.Hit/(r2dh.atBat+0.0001) AS BA_home,
       r2da.Hit/(r2da.atBat+0.0001) AS BA_away,
       r2dh.atBat/(r2dh.Home_Run+0.0001) AS A_to_HR_home,
       r2da.atBat/ (r2da.Home_Run+0.0001) AS A_to_HR_away,
       r2dh.Home_Run/(r2dh.Hit+0.0001)  AS HR_per_hit_home,
       r2da.Home_Run/(r2da.Hit+0.0001) AS HR_per_hit_away,
       (r2dh.Hit - r2dh.Home_Run)/(r2dh.atBat - r2dh.Strikeout - r2dh.Home_Run + r2dh.Sac_Fly +0.0001) AS BABIP_home,
       (r2da.Hit - r2da.Home_Run)/(r2da.atBat - r2da.Strikeout - r2da.Home_Run + r2da.Sac_Fly +0.0001) AS BABIP_away,
       r2dh.Ground_Out/(r2dh.Fly_Out+0.0001) AS GO_to_FO_home,
       r2da.Ground_Out/(r2da.Fly_Out+0.0001) AS GO_to_FO_away,
       r2dh.Walk/(r2dh.Strikeout+0.0001) AS W_to_SO_home,
       r2da.Walk/(r2da.Strikeout+0.0001) AS W_to_SO_away,
       r2dh.plateAppearance/(r2dh.Strikeout+0.0001) AS PA_to_SO_home,
       r2da.plateAppearance/(r2da.Strikeout +0.0001) AS PA_to_SO_away,
       r2dh.Hit + r2dh.`Double` + r2dh.Triple*2 + r2dh.Home_Run*3 AS TB_home,
       r2da.Hit + r2da.`Double` + r2da.Triple*2 + r2da.Home_Run*3 AS TB_away,
       r2dh.Hit + r2dh.Walk + r2dh.Hit_By_Pitch AS TOB_home,
       r2da.Hit + r2da.Walk + r2da.Hit_By_Pitch AS TOB_away,
       r2dh.`Double` + r2dh.Triple + r2dh.Home_Run AS XBH_home,
       r2da.`Double` + r2da.Triple + r2da.Home_Run AS XBH_away,
       (r2dh.Hit + r2dh.Walk + r2dh.Hit_By_Pitch)/(r2dh.atBat + r2dh.Walk + r2dh.Hit_By_Pitch + r2dh.Sac_Fly +0.0001) AS OBP_home,
       (r2da.Hit + r2da.Walk + r2da.Hit_By_Pitch)/(r2da.atBat + r2da.Walk + r2da.Hit_By_Pitch + r2da.Sac_Fly +0.0001) AS OBP_away,
       (r2dh.Single + r2dh.`Double`*2 + r2dh.Triple*3 + r2dh.Home_Run*4)/(r2dh.atBat+0.0001) AS SLG_home,
       (r2da.Single + r2da.`Double`*2 + r2da.Triple*3 + r2da.Home_Run*4)/(r2da.atBat+0.0001) AS SLG_away
FROM game g
JOIN rolling_200_day r2dh ON g.game_id = r2dh.game_id AND g.home_team_id = r2dh.team_id
JOIN rolling_200_day r2da ON g.game_id = r2da.game_id AND g.away_team_id = r2da.team_id;  


# adding extra features

CREATE TABLE IF NOT EXISTS baseball_features_2 AS
SELECT game_id AS gid, home_team_id AS hid, away_team_id AS aid,
       OBP_home + SLG_home AS OPS_home,
       OBP_away + SLG_away AS OPS_away,
       (OBP_home * 1.8 + SLG_home)/4 AS GPA_home,
       (OBP_away * 1.8 + SLG_away)/4 AS GPA_away,
       SLG_home - BA_home AS ISO_home,
       SLG_away - BA_away AS ISO_away
FROM baseball_features;

# merging the two feature tables

CREATE TABLE IF NOT EXISTS baseball_features_3 AS 
SELECT * 
FROM baseball_features   bsf
JOIN baseball_features_2 bsf2
ON   bsf.game_id = bsf2.gid;


CREATE UNIQUE INDEX id ON baseball_features_3(game_id);

# creating game_results table

CREATE TABLE IF NOT EXISTS game_results AS
SELECT game_id AS gmid, 
       CASE 
          WHEN winner_home_or_away = 'H' THEN 1
          WHEN winner_home_or_away = 'A' THEN 0
          ELSE 0 END AS home_team_wins
FROM boxscore;

# creating intermediary predictor table

CREATE TABLE IF NOT EXISTS res_pred AS
SELECT * 
FROM baseball_features_3 bs3
JOIN game_results        gr
ON   bs3.game_id = gr.gmid
ORDER BY bs3.game_id;

# dropping redundant columns

ALTER TABLE res_pred 
DROP COLUMN gmid, 
DROP COLUMN gid, 
DROP COLUMN hid, 
DROP COLUMN aid;

# creating final features table

CREATE TABLE IF NOT EXISTS final_baseball_features AS
SELECT 
     game_id, home_team_id, away_team_id,
     BA_home - BA_away AS BA_diff,
     BA_home/(BA_away+0.0001)  AS BA_ratio,
     A_to_HR_home - A_to_HR_away AS A_to_HR_diff,
     A_to_HR_home/(A_to_HR_away+0.0001) AS A_to_HR_ratio,
     HR_per_hit_home - HR_per_hit_away AS HR_per_hit_diff,
     HR_per_hit_home/(HR_per_hit_away+0.0001) AS HR_per_hit_ratio,
     BABIP_home - BABIP_away AS BABIP_diff,
     BABIP_home/(BABIP_away+0.0001) AS BABIP_ratio,
     GO_to_FO_home - GO_to_FO_away AS GO_to_FO_diff,
     GO_to_FO_home/(GO_to_FO_away+0.0001) AS GO_to_FO_ratio,
     W_to_SO_home - W_to_SO_away AS W_to_SO_diff,
     W_to_SO_home/(W_to_SO_away+0.0001) AS W_to_SO_ratio,
     PA_to_SO_home - PA_to_SO_away AS PA_to_SO_diff,
     PA_to_SO_home/(PA_to_SO_away+0.0001) AS PA_to_SO_ratio,
     TB_home - TB_away AS TB_diff,
     TB_home/(TB_away+0.0001) AS TB_ratio,
     TOB_home - TOB_away AS TOB_diff,
     TOB_home/(TOB_away+0.0001) AS TOB_ratio,
     XBH_home - XBH_away AS XBH_diff,
     XBH_home/(XBH_away+0.0001) AS XBH_ratio,
     OBP_home - OBP_away AS OBP_diff,
     OBP_home/(OBP_away+0.0001) AS OBP_ratio,
     SLG_home - SLG_away AS SLG_diff,
     SLG_home/(SLG_away+0.0001) AS SLG_ratio,
     OPS_home - OPS_away AS OPS_diff,
     OPS_home/(OPS_away+0.0001) AS OPS_ratio,
     GPA_home -  GPA_away AS  GPA_diff,
     GPA_home/(GPA_away+0.0001) AS  GPA_ratio,
     ISO_home - ISO_away AS ISO_diff,
     ISO_home/(ISO_away+0.0001) AS ISO_ratio,
     home_team_wins
FROM res_pred
ORDER BY game_id;

     
     


     




        










