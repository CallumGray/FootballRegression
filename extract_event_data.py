import json
import os

base_path = 'C:\\Users\\Callum\\Documents\\GitHub\\open-data\\data'
season_id = 27 # 2015/2016 season

# # Find the competition ids
# competitions_file = os.path.join(base_path, 'competitions.json')
# competition_ids = {}
#
# with open(competitions_file, 'r', encoding='utf-8') as f:
#     competitions_json = json.load(f)
#
# for comp in competitions_json:
#     if comp['season_name'] == '2015/2016':
#         competition_ids[comp['competition_id']] = comp['competition_name']
#
# print(competition_ids)

competition_ids = {9:'bundesliga', 11:'la-liga', 7:'ligue-1', 2:'premier-league', 12:'serie-a'}

# Create a dictionary of competition_id:[matches]
match_ids_by_competition = {}

for competition_id in competition_ids.keys():
    # File of all
    matches_file = os.path.join(base_path, "matches", str(competition_id), str(season_id)+'.json')

    with open(matches_file, 'r', encoding='utf-8') as f:
        matches_json = json.load(f)

    for match in matches_json:
        if competition_id not in match_ids_by_competition:
            match_ids_by_competition[competition_id] = []

        match_ids_by_competition[competition_id].append(match['match_id'])

# for key, value in match_ids_by_competition.items():
#     print('{} ({} matches)'.format(competition_ids[key], len(value)))
#     print(value)
#     print()

for competition_id, competition_name in competition_ids.items():

    print(competition_name,':')

    all_shots = []
    all_assists = []
    total_matches = len(match_ids_by_competition[competition_id])

    for i, match_id in enumerate(match_ids_by_competition[competition_id]):
        events_file = os.path.join(base_path, 'events', str(match_id)+'.json')

        with open(events_file, 'r', encoding='utf-8') as f:
            events_json = json.load(f)

        shots = [event for event in events_json if event['type']['name'] == 'Shot']
        assists = [event for event in events_json if event['type']['name'] == 'Pass' and 'assisted_shot_id' in event['pass']]
        all_shots += shots
        all_assists += assists
        if (i+1) % 50 == 0:
            print(i+1,'/',total_matches)

    # Output all shots and passes to json files
    directory = 'Data'
    shots_filename = os.path.join(directory, competition_name + '-shots.json')
    assists_filename = os.path.join(directory, competition_name + '-assists.json')

    with open(shots_filename, 'w', encoding='utf-8') as file:
        json.dump(all_shots, file, indent=4, ensure_ascii=False)

    with open(assists_filename, 'w', encoding='utf-8') as file:
        json.dump(all_assists, file, indent=4, ensure_ascii=False)

    print('Done!')
    print()