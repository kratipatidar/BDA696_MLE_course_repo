-- LISTING databases

show databases;

-- using BASEBALL database

use baseball;

-- listing tables

show tables;

-- prodding around the databse, looking at tables

select * from game 
limit 0, 100;

select * from atbat_r
limit 0, 20;

select * from batter_counts 
limit 0, 20;

select * from battersingame 
limit 0, 20;

select * from hits
limit 0, 20;

-- creating a new table by joining batter_counts and game
create table bc_join_g AS
select g.game_id, bc.batter, bc.Hit, bc.atBat, g.local_date 
from batter_counts bc 
join game          g
on   bc.game_id = g.game_id ;

-- checking the new table
select * from bc_join_g;

-- annual batting average (BA) of each player
create or replace table annual_ba as
select bc.batter, sum(Hit)/sum(atBat+ 0.00001) as BA, Year(local_date) as year -- added 0.00001 to prevent division by zero error
from batter_counts bc 
join game          g
on   bc.game_id = g.game_id
group by bc.batter, year(local_date)
order by year desc ;

select * from annual_ba 
limit 0,50;


-- historic batting average (BA) of each player
create or replace table historic_ba as
select bc.batter, sum(Hit)/sum(atBat + 0.00001) as BA -- added 0.00001 to prevent division by zero error
from batter_counts bc 
join game          g
on   bc.game_id = g.game_id
group by bc.batter
order by BA asc ;

-- drop table historic_ba;

select * from historic_ba 
limit 0, 50;

-- rolling average

create or replace table rolling_ba_100_days as
select a.game_id, a.batter, sum(b.Hit)/sum(b.atBat) as BA, count(*) as cnt, b.local_date
from   bc_join_g a
join   bc_join_g b
on a.batter = b.batter
and b.local_date > date_sub(a.local_date, interval 100 day) and a.local_date > b.local_date
where a.game_id = 10000       -- comment out this line to get the rolling batting average for all the players across all games
group by a.game_id, a.batter, a.local_date;  

select * from rolling_ba_100_days;    



      
      
      



