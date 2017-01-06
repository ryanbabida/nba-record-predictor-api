import glob
import numpy as np 

def load_data():
	teams = [   'Atlanta Hawks\n',
				'Boston Celtics\n',
				'Brooklyn Nets\n',
				'Charlotte Bobcats\n',
				'Charlotte Hornets\n',
				'Chicago Bulls\n',
				'Cleveland Cavaliers\n',
				'Dallas Mavericks\n',
				'Denver Nuggets\n',
				'Detroit Pistons\n',
				'Golden State Warriors\n',
				'Houston Rockets\n',
				'Indiana Pacers\n',
				'LA Clippers\n',
				'Los Angeles Clippers\n',
				'Los Angeles Lakers\n',
				'Memphis Grizzlies\n',
				'Miami Heat\n',
				'Milwaukee Bucks\n',
				'Minnesota Timberwolves\n',
				'New Jersey Nets\n',
				'New Orleans Hornets\n',
				'New Orleans Pelicans\n',
				'New York Knicks\n',
				'Oklahoma City Thunder\n',
				'Orlando Magic\n',
				'Philadelphia 76ers\n',
				'Phoenix Suns\n',
				'Portland Trail Blazers\n',
				'Sacramento Kings\n',
				'San Antonio Spurs\n',
				'Seattle SuperSonics\n',
				'Toronto Raptors\n',
				'Utah Jazz\n',
				'Vancouver Grizzlies\n',
				'Washington Bullets\n',
				'Washington Wizards\n',
	]
	# Create data file
	data = open('data_processing/data/data.csv', 'w+')
	data.write("GP,W,L,WIN%,MIN,PTS,FGM,FGA,FG%,3PM,3PA,3P%,FTM,FTA,FT%,OREB,DREB,REB,AST,TOV,STL,BLK,BLKA,PF,PFD,+/-\n")
	data = open('data_processing/data/data.csv', 'a+')

	# Push all the statistics from seasons 1997-1998 into one file for analysis
	for filename in glob.glob('data_processing/data/' + '*-*.csv'):
		print(filename)
		inf = open(filename, 'r')
		for line in inf.readlines():
			if line not in teams and 'GP' not in line and line != '' and len(line) > 5: 
				data.write(line)
		data.write('\n')
	data.close()

	data = np.genfromtxt('data_processing/data/data.csv', dtype=float, delimiter=',', names=True)

	features = np.matrix([data['MIN'], data['PTS'], data['FGM'], data['FGA'], data['FG'], data['3PM'], data['3PA'], 
				data['3P'], data['FTM'], data['FTA'], data['FT'], data['OREB'], data['DREB'], 
				data['REB'], data['AST'], data['TOV'], data['STL'], data['BLK'], data['BLKA'], 
				data['PF'], data['PFD'], data['f0']]).T
	target = np.matrix(data['WIN'])
	target = target.T

	return features, target, data.dtype.names

3
def load_test():
	teams = [   'Atlanta Hawks\n',
				'Boston Celtics\n',
				'Brooklyn Nets\n',
				'Charlotte Bobcats\n',
				'Charlotte Hornets\n',
				'Chicago Bulls\n',
				'Cleveland Cavaliers\n',
				'Dallas Mavericks\n',
				'Denver Nuggets\n',
				'Detroit Pistons\n',
				'Golden State Warriors\n',
				'Houston Rockets\n',
				'Indiana Pacers\n',
				'LA Clippers\n',
				'Los Angeles Clippers\n',
				'Los Angeles Lakers\n',
				'Memphis Grizzlies\n',
				'Miami Heat\n',
				'Milwaukee Bucks\n',
				'Minnesota Timberwolves\n',
				'New Jersey Nets\n',
				'New Orleans Hornets\n',
				'New Orleans Pelicans\n',
				'New York Knicks\n',
				'Oklahoma City Thunder\n',
				'Orlando Magic\n',
				'Philadelphia 76ers\n',
				'Phoenix Suns\n',
				'Portland Trail Blazers\n',
				'Sacramento Kings\n',
				'San Antonio Spurs\n',
				'Seattle SuperSonics\n',
				'Toronto Raptors\n',
				'Utah Jazz\n',
				'Vancouver Grizzlies\n',
				'Washington Bullets\n',
				'Washington Wizards\n',
	]
	# Create data file
	data = open('data_processing/data/test.csv', 'w+')
	data.write("GP,W,L,WIN%,MIN,PTS,FGM,FGA,FG%,3PM,3PA,3P%,FTM,FTA,FT%,OREB,DREB,REB,AST,TOV,STL,BLK,BLKA,PF,PFD,+/-\n")
	data = open('data_processing/data/test.csv', 'a+')

	team_names = list()

	# Push all the statistics from seasons 1997-1998 into one file for analysis
	for filename in glob.glob('data_processing/data/' + 'testset.csv'):
		print(filename)
		inf = open(filename, 'r')
		for line in inf.readlines():
			if line not in teams and 'GP' not in line and line != '' and len(line) > 5: 
				data.write(line)
			if line in teams: 
				team_names.append(line.strip('\n'))
		data.write('\n')
	data.close()

	data = np.genfromtxt('data_processing/data/test.csv', dtype=float, delimiter=',', names=True)

	features = np.matrix([data['MIN'], data['PTS'], data['FGM'], data['FGA'], data['FG'], data['3PM'], data['3PA'], 
				data['3P'], data['FTM'], data['FTA'], data['FT'], data['OREB'], data['DREB'], 
				data['REB'], data['AST'], data['TOV'], data['STL'], data['BLK'], data['BLKA'], 
				data['PF'], data['PFD'], data['f0']]).T
	target = np.matrix(data['WIN'])
	target = target.T
	games_played = data['GP']
	wins = data['W']

	return features, target, data.dtype.names, team_names, games_played, wins
