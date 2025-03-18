import json
import os
import numpy as np


def points_in_triangle(points, a, b, c):
    area_abc = 0.5 * np.abs(
        a[0] * (b[1] - c[1]) +
        b[0] * (c[1] - a[1]) +
        c[0] * (a[1] - b[1])
    )

    area_abd = 0.5 * np.abs(
        a[0] * (b[1] - points[:, 1]) +
        b[0] * (points[:, 1] - a[1]) +
        points[:, 0] * (a[1] - b[1])
    )

    area_bcd = 0.5 * np.abs(
        b[0] * (c[1] - points[:, 1]) +
        c[0] * (points[:, 1] - b[1]) +
        points[:, 0] * (b[1] - c[1])
    )

    area_cad = 0.5 * np.abs(
        c[0] * (a[1] - points[:, 1]) +
        a[0] * (points[:, 1] - c[1]) +
        points[:, 0] * (c[1] - a[1])
    )

    in_triangle = np.isclose(area_abc, area_abd + area_bcd + area_cad)
    return np.sum(in_triangle)


def angle_abc(a, b, c):
    # Cosine rule to find angle
    BA = a - b
    BC = c - b
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    return np.arccos(cos_theta)


left_goalpost = np.array([120, 36])
right_goalpost = np.array([120, 44])
goal_center = np.array([120, 40])
attacking_positions = ['Right Wing', 'Right Attacking Midfield', 'Center Attacking Midfield', 'Left Attacking Midfield',
                       'Left Wing', 'Right Center Forward', 'Striker', 'Left Center Forward', 'Secondary Striker']

competition_ids = {9:'bundesliga', 11:'la-liga', 7:'ligue-1', 2:'premier-league', 12:'serie-a'}
competition_names = competition_ids.values()

for competition_name in competition_names:

    print(competition_name,':')

    directory = 'Data'
    shots_filename = os.path.join(directory, competition_name + '-shots.json')
    assists_filename = os.path.join(directory, competition_name + '-assists.json')
    output_filename = os.path.join(directory,competition_name+'.json')

    with open(shots_filename, 'r', encoding='utf-8') as f:
        all_shots = json.load(f)

    with open(assists_filename, 'r', encoding='utf-8') as f:
        all_assists = json.load(f)

    final_shots = []

    # Dictionary of {shot_id: 'Ground Pass'} as an example
    # Used for assist data
    assist_lookup = {}

    total_assists = len(all_assists)-1
    for index, assist_event in enumerate(all_assists):
        assist_lookup[assist_event['pass']['assisted_shot_id']] = assist_event['pass']['height']['name']

    total_shots = len(all_shots) - 1
    for index, shot_event in enumerate(all_shots):

        final_shot = {}

        if index % 250 == 0:
            print(index, '/', total_shots)

        # Important shot_event data
        final_shot['index'] = index
        final_shot['id'] = shot_event['id']
        final_shot['team'] = shot_event['team']['name']
        final_shot['player'] = shot_event['player']['name']

        shot_data = shot_event['shot']

        shot_location = None
        keeper_location = [119.9, 40] # If the keeper is off the pitch (inside his goal) he won't be recorded, so mark him as there
        team_locations = []
        op_locations = []

        if 'freeze_frame' not in shot_data:
            # No freeze_frame data is given for penalties
            shot_location = [108, 40]
            keeper_location = [119.9, 40]
            op_locations.append(keeper_location)
        else:
            freeze_frame = shot_data['freeze_frame']
            shot_location = shot_event['location']

            for player in freeze_frame:
                player_location = player['location']

                if player['teammate']:
                    team_locations.append(player_location)
                else:
                    # Goalkeeper
                    if player['position']['id'] == 1:
                        keeper_location = player_location

                    op_locations.append(player_location)

        # Locations, for use when plotting the overhead image of a shot attempt
        final_shot['shot_location'] = shot_location
        final_shot['keeper_location'] = keeper_location
        final_shot['team_locations'] = team_locations
        final_shot['op_locations'] = op_locations

        # Convert to numpy arrays for calculations
        shot_location = np.array(shot_location)
        team_locations = np.array(team_locations)
        op_locations = np.array(op_locations)
        # Convert this one to a list so it can be used in the points_in_triangle function
        keeper_location = np.array([keeper_location])

        # Professionally calculated xg. Will be compared after building the model.
        final_shot['statsbomb_xg'] = shot_data['statsbomb_xg']

        # Distance from the shot location to the center of the goal
        final_shot['distance'] = float(np.linalg.norm(shot_location - goal_center))

        # Angle at the shot location to the 2 goalposts
        final_shot['angle'] = float(angle_abc(left_goalpost, shot_location, right_goalpost))

        # Is the keeper standing within the above angle
        final_shot['keeper_in_position'] = int(points_in_triangle(keeper_location, left_goalpost, shot_location, right_goalpost))

        # How many opposing bodies total are in the above angle
        final_shot['blocking_bodies'] = int(points_in_triangle(op_locations, left_goalpost, shot_location, right_goalpost))

        # Number of surrounding players (2 yards)
        op_distances = np.linalg.norm(op_locations - shot_location, axis=1)
        final_shot['pressuring_players'] = int(np.sum(op_distances <= 2))

        # Player is a winger/striker/attacking mid
        final_shot['attacking_position'] = 1 if shot_event['position']['name'] in attacking_positions else 0

        # How high the ball was during the assist (will all be 0 for no assist)
        final_shot['assist_ground'] = 0
        final_shot['assist_low'] = 0
        final_shot['assist_high'] = 0

        if shot_event['id'] in assist_lookup:
            final_shot['assist_ground'] = 1 if assist_lookup[shot_event['id']] == 'Ground Pass' else 0
            final_shot['assist_low'] = 1 if assist_lookup[shot_event['id']] == 'Low Pass' else 0
            final_shot['assist_high'] = 1 if assist_lookup[shot_event['id']] == 'High Pass' else 0

        # 'Open Play' or 'Corner' given by all 0
        final_shot['type_penalty'] = 1 if shot_data['type']['name'] == 'Penalty' else 0
        final_shot['type_freekick'] = 1 if shot_data['type']['name'] == 'Free Kick' else 0

        #  'Normal' given by all 0
        final_shot['technique_lob'] = 1 if shot_data['technique']['name'] == 'Lob' else 0
        final_shot['technique_halfvolley'] = 1 if shot_data['technique']['name'] == 'Half Volley' else 0
        final_shot['technique_volley'] = 1 if shot_data['technique']['name'] == 'Volley' else 0
        final_shot['technique_other'] = 1 if shot_data['technique']['name'] in ['Overhead Kick', 'Backheel', 'Diving Header'] else 0

        # 'Left Foot' or 'Right Foot' given by all 0
        final_shot['body_head'] = 1 if shot_data['body_part']['name'] in ['Head'] else 0
        final_shot['body_other'] = 1 if shot_data['body_part']['name'] in ['Other'] else 0

        # 'Normal' (and a few other niche play patterns) given by all 0
        final_shot['pattern_corner'] = 1 if shot_event['play_pattern']['name'] == 'From Corner' else 0
        final_shot['pattern_freekick'] = 1 if shot_event['play_pattern']['name'] == 'From Free Kick' else 0
        final_shot['pattern_throw'] = 1 if shot_event['play_pattern']['name'] == 'From Throw In' else 0
        final_shot['pattern_counter'] = 1 if shot_event['play_pattern']['name'] == 'From Counter' else 0

        # OUTCOME
        final_shot['goal'] = 1 if shot_data['outcome']['name'] == 'Goal' else 0

        final_shots.append(final_shot)

    print('Done!')
    print()

    # Output all shots to a json file
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(final_shots, file, indent=4, ensure_ascii=False)

