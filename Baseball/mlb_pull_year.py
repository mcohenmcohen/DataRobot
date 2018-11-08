import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib
import csv
# import os
import datetime
import time
import argparse as ap

base_url = "http://gd2.mlb.com/components/game/mlb/"
game_dict = {}
player_names = {}
atbats = []
pitches = []
home_catchers = []
away_catchers = []
home_pitchers = []
away_pitchers = []
debug = 0
test = 0

def parse_game(url):
    while True:
        try:
            xmlfile = BeautifulSoup(urlopen(url), "xml")
            break
        except urllib.error.URLError:
            time.sleep(10)
            pass
        pass

    if debug > 0:
        print("Parsed game")
        pass

    if 'type' in xmlfile.game.attrs:
        game_dict['game_type'] = xmlfile.game["type"]
    else:
        game_dict['game_type'] = "U"
        pass
    if game_dict['game_type'] == "S":
        game_dict['game_type_des'] = "Spring Training"

        game_dict['st_fl'] = "T"
    elif game_dict['game_type'] == "R":
        game_dict['game_type_des'] = "Regular Season"
        game_dict['regseason_fl'] = "T"
    elif game_dict['game_type'] == "F":
        game_dict['game_type_des'] = "Wild-card Game"
        game_dict['playoff_fl'] = "T"
    elif game_dict['game_type'] == "D":
        game_dict['game_type_des'] = "Divisional Series"
        game_dict['playoff_fl'] = "T"
    elif game_dict['game_type'] == "L":
        game_dict['game_type_des'] = "LCS"
        game_dict['playoff_fl'] = "T"
    elif game_dict['game_type'] == "W":
        game_dict['game_type_des'] = "World Series"
        game_dict['playoff_fl'] = "T"
    else:
        game_dict['game_type_des'] = "Unknown"
        pass

    if 'local_game_time' in xmlfile.game.attrs:
        game_dict['local_game_time'] = xmlfile.game["local_game_time"]
    else:
        game_dict['local_game_time'] = "unknown"
        pass

    if 'game_pk' in xmlfile.game.attrs:
        game_dict['game_id'] = xmlfile.game["game_pk"]
    else:
        game_dict['game_id'] = "unknown"
        pass

    if xmlfile.find("team"):
        game_dict['home_team_id'] = xmlfile.find("team", type="home")["code"]
        game_dict['away_team_id'] = xmlfile.find("team", type="away")["code"]
        game_dict['home_team_lg'] = xmlfile.find("team", type="home")["league"]
        game_dict['away_team_lg'] = xmlfile.find("team", type="away")["league"]
    else:
        game_dict['home_team_id'] = "unknown"
        game_dict['away_team_id'] = "unknown"
        game_dict['home_team_lg'] = "unknown"
        game_dict['away_team_lg'] = "unknown"
        pass

    if game_dict['home_team_lg'] == game_dict['away_team_lg']:
        game_dict['interleague_fl'] = "F"
    else:
        game_dict['interleague_fl'] = "T"
        pass

    if xmlfile.find("stadium"):
        game_dict['park_id'] = xmlfile.stadium["id"]
        game_dict['park_name'] = xmlfile.stadium["name"]
        game_dict['park_loc'] = xmlfile.stadium["location"]
    else:
        game_dict['park_id'] = "unknown"
        game_dict['park_name'] = "unknown"
        game_dict['park_loc'] = "unknown"
        pass
    return


def grab_catchers(xmlfile, teamflag):
    catchers = []
    team = xmlfile.find("batting", team_flag=teamflag)
    for batter in team.children:
        if batter.name == "batter":
            pos = batter["pos"]
            player_names[batter["id"]] = batter["name_display_first_last"]
            count = 1
            for p in re.split('-', pos):
                if p == "C":
                    catcher = {}
                    catcher["id"] = batter["id"]
                    catcher["name"] = batter["name_display_first_last"]
                    if int(batter["bo"]) % 100 == 0:
                        if count == 1:
                            catcher["type"] = "starter"
                            if teamflag == "home":
                                game_dict["curr_home_catcher_id"] = catcher["id"]
                                game_dict["curr_home_catcher_name"] = catcher["name"]
                                pass
                            else:
                                game_dict["curr_away_catcher_id"] = catcher["id"]
                                game_dict["curr_away_catcher_name"] = catcher["name"]
                                pass
                        else:
                            catcher["type"] = "reserve"
                            pass
                    else:
                        catcher["type"] = "reserve"
                        pass
                    catchers.append(catcher.copy())
                    pass
                count += 1
                pass
            pass
        pass

    return catchers


def grab_pitchers(xmlfile, teamflag):
    pitchers = []
    team = xmlfile.find("pitching", team_flag=teamflag)
    count = 1
    for pitch in team.children:
        if pitch.name == "pitcher":
            player_names[pitch["id"]] = pitch["name_display_first_last"]
            pitcher = {}
            pitcher["id"] = pitch["id"]
            name = pitch["name_display_first_last"]
            name2 = re.sub(r'(\w\.)(\w)', r'\1 \2', name)
            if debug > 2:
                print("{} to {}".format(name, name2))
                pass
            pitcher["name"] = name
            if count == 1:
                pitcher["type"] = "starter"
                if teamflag == "home":
                    game_dict["curr_home_pitcher_id"] = pitcher["id"]
                    game_dict["curr_home_pitcher_name"] = pitcher["name"]
                    pass
                else:
                    game_dict["curr_away_pitcher_id"] = pitcher["id"]
                    game_dict["curr_away_pitcher_name"] = pitcher["name"]
                    pass
            else:
                pitcher["type"] = "reserve"
                pass
            pitchers.append(pitcher.copy())
            count += 1
            pass
        pass

    return pitchers


def parse_boxscore(url):
    global home_catchers, away_catchers
    global home_pitchers, away_pitchers
    while True:
        try:
            xmlfile = BeautifulSoup(urlopen(url), "xml")
            break
        except urllib.error.URLError:
            time.sleep(10)
            pass
        pass

    if debug > 0:
        print("Parse boxscore")
        pass

    home_catchers = grab_catchers(xmlfile, "home")
    away_catchers = grab_catchers(xmlfile, "away")
    home_pitchers = grab_pitchers(xmlfile, "home")
    away_pitchers = grab_pitchers(xmlfile, "away")
    return


def parse_players(url):
    while True:
        try:
            xmlfile = BeautifulSoup(urlopen(url), "xml")
            break
        except urllib.error.URLError:
            time.sleep(10)
            pass
        pass
    if debug > 0:
        print("Parsed players")
        pass

    umps = xmlfile.find("umpires")
    found_hp = 0
    for ump in umps.children:
        if ump.name == "umpire":
            if ump["position"] == "home":
                if found_hp == 0:
                    game_dict["hp_umpire_id"] = ump["id"]
                    game_dict["hp_umpire_name"] = ump["name"]
                    found_hp = 1
                else:
                    print("ERR: Found multiple home plate umpires in {}".format(url))
                    pass
                pass
            elif re.search("home", ump["position"]):
                print("ERR: Found unexpected position {} in {}".format(ump["position"], url))
                pass
            pass
        pass
    return


def parse_rawBoxscore(url):
    """Could grab the umpire from here and get additional information on venue, start time, weather and wind"""
    while True:
        try:
            xmlfile = BeautifulSoup(urlopen(url), "xml")
            break
        except urllib.error.URLError:
            time.sleep(10)
            pass
        pass
    if debug > 0:
        print("Parsed raw boxscore")
        pass

    if 'venue_name' in xmlfile.boxscore.attrs:
        game_dict["venue_name"] = xmlfile.boxscore['venue_name']
        pass
    if 'start_time' in xmlfile.boxscore.attrs:
        game_dict["start_time"] = xmlfile.boxscore['start_time']
        pass
    if 'weather' in xmlfile.boxscore.attrs:
        game_dict["weather"] = xmlfile.boxscore['weather']
        pass
    if 'wind' in xmlfile.boxscore.attrs:
        game_dict["wind"] = xmlfile.boxscore['wind']
        pass
    if xmlfile.boxscore.find("umpires"):
        umps = xmlfile.boxscore.find("umpires")
        for ump in umps.children:
            if 'position' in ump.attrs:
                if 'HP' == ump['position']:
                    if 'id' in ump.attrs:
                        game_dict["umpire"] = ump['id']
                        pass
                    pass
                pass
            pass
        pass
    return


def parse_action(action, torb, inatbat):
    if debug > 1:
        print("action: {} - {}".format(action["event"], action["des"]), flush=True)
        pass
    if ((re.search('defensive sub', action["event"], re.I) and
         re.search('playing catcher', action["des"], re.I)) or
        (re.search('defensive switch', action["event"], re.I) and
         (re.search('to catcher', action["des"], re.I) or
         (re.search('as catcher', action["des"], re.I))))):
        if debug > 1:
            print("Found catcher change: {} - {}".format(action["event"], action["des"]), flush=True)
            pass
        catchers = home_catchers
        if torb == "bottom":
            catchers = away_catchers
            pass
        name = None
        id = None
        found = 0
        for c in catchers:
            if c["id"] == action["player"]:
                found = 1
                id = action["player"]
                name = c["name"]
                if c["type"] != "reserve":
                    print("ERR: New catcher {} not a reserve".format(name))
                    pass
                pass
            pass
        if found == 0:
            print("ERR: New catcher {} not found".format(action["player"]))
            pass
        else:
            if debug > 1:
                print("New catcher {} found".format(name), flush=True)
                pass

            if torb == "top":
                game_dict["curr_home_catcher_id"] = id
                game_dict["curr_home_catcher_name"] = name
                pass
            else:
                game_dict["curr_away_catcher_id"] = id
                game_dict["curr_away_catcher_name"] = name
                pass
            pass
        pass
    elif (re.search('pitching substitution', action["event"], re.I) or
          (re.search('defensive switch', action["event"], re.I) and
           re.search('to pitcher', action["des"], re.I))):
        if debug > 1:
            print("Found pitching change", flush=True)
            pass
        pitchers = home_pitchers
        if torb == "bottom":
            pitchers = away_pitchers
            pass
        name = None
        id = None
        if debug > 1:
            print("Matching: {}".format(action["des"]), flush=True)
            pass
#        match = re.search(r':\s+((\S+\s*)+)\s+replaces', action["des"], re.I)
        match = re.search(r':\s+(.+)\s+replaces', action["des"], re.I)
        if debug > 1:
            print("Match finished", flush=True)
            pass
        m_name = None
        m_id = None
        if match:
            m_name = match.group(1)
            if debug > 0:
                print("Match made: {}".format(action["des"]), flush=True)
                print(m_name)
                pass
            pass
        else:
            match = re.search(r'pitcher\s+(.+)\s+enters the batting order',
                              action["des"], re.I)
            if match:
                m_name = match.group(1)
                if debug > 0:
                    print("Match made: {}".format(action["des"]), flush=True)
                    print(m_name)
                    pass
                pass
            else:
                print("Match not made: {}".format(action["des"]), flush=True)
                pass
            pass

        found = 0
        found_name = 0
        if debug > 1:
            print("Checking pitcher", flush=True)
            pass
        for p in pitchers:
            if p["id"] == action["player"]:
                found = 1
                id = action["player"]
                name = p["name"]
                if p["type"] != "reserve":
                    print("ERR: New pitcher {} not a reserve".format(name))
                    pass
                pass
            if p["name"] == m_name:
                found_name = 1
                m_id = p["id"]
                pass
            pass

        if found_name == 1:
            if found == 1:
                if id == m_id:
                    if name == m_name:
                        pass
                    else:
                        print("ERR: Mismatched pitcher names: {} and {} for {}, {}".format(name, m_name, id, m_id))
                        pass
                    pass
                elif id == 0:
                    if m_id:
                        id = m_id
                        pass
                    name = m_name
                    found = 1
                    pass
                else:
                    print("WARN: Mismatched pitcher names: {} and {} for {}, {}".format(name, m_name, id, m_id))
                    if m_id:
                        id = m_id
                        pass
                    name = m_name
                    found = 1
                    pass
                pass
            else:
                if m_id:
                    id = m_id
                    pass
                name = m_name
                found = 1
                pass
            pass

        if found == 0:
            print("ERR: New pitcher {} not found".format(action["player"]))
            pass
        else:
            if torb == "top":
                game_dict["curr_home_pitcher_id"] = id
                game_dict["curr_home_pitcher_name"] = name
                pass
            else:
                game_dict["curr_away_pitcher_id"] = id
                game_dict["curr_away_pitcher_name"] = name
                pass
            pass
        pass
    elif (inatbat and
          (re.search('batter', action["event"], re.I) or
           re.search('batter', action["des"], re.I))):
        print("ERR: Batter action in at bat: {}, {}".format(action["event"], action["des"]))
    return


def parse_atbat(ab):
    atbat_dict = {}

    atbat_dict["battedball_cd"] = ""
    base1 = "_"
    base2 = "_"
    base3 = "_"

    if 'b' in ab.attrs:
        atbat_dict["ball_ct"] = ab["b"]
        pass
    else:
        atbat_dict["ball_ct"] = ""
        pass

    if 's' in ab.attrs:
        atbat_dict["strike_ct"] = ab["s"]
        pass
    else:
        atbat_dict["strike_ct"] = ""
        pass

    if 'o' in ab.attrs:
        atbat_dict["event_outs_ct"] = ab["o"]
        pass
    else:
        atbat_dict["event_outs_ct"] = ""
        pass

    atbat_dict["bat_home_id"] = 0
    if 'batter' in ab.attrs:
        atbat_dict["bat_mlbid"] = ab["batter"]
        atbat_dict["bat_name"] = player_names[ab["batter"]]
        pass
    else:
        atbat_dict["bat_mlbid"] = ""
        pass

    if 'stand' in ab.attrs:
        atbat_dict["bat_hand_cd"] = ab["stand"]
        pass
    else:
        atbat_dict["bat_hand_cd"] = ""
        pass

    if 'pitcher' in ab.attrs:
        atbat_dict["pit_mlbid"] = ab["pitcher"]
        atbat_dict["pit_name"] = player_names[ab["pitcher"]]
        pass
    else:
        atbat_dict["pit_mlbid"] = ""
        pass

    if 'p_throws' in ab.attrs:
        atbat_dict["pit_hand_cd"] = ab["p_throws"]
        pass
    else:
        atbat_dict["pit_hand_cd"] = ""
        pass

    if 'des' in ab.attrs:
        ab_des = ab["des"]
        pass
    else:
        ab_des = ""
        pass
    atbat_dict["ab_des"] = ab_des

    if 'num' in ab.attrs:
        atbat_dict["ab_number"] = ab["num"]
        pass
    else:
        atbat_dict["ab_number"] = ""
        pass

    if 'event' in ab.attrs:
        event_tx = ab["event"]
        pass
    else:
        event_tx = ""
        pass
    atbat_dict["event_tx"] = event_tx

    atbat_dict["event_cd"] = ""
    if (event_tx == "Flyout" or event_tx == "Fly Out" or
        event_tx == "Sac Fly" or event_tx == "Sac Fly DP"):
        atbat_dict["event_cd"] = 2
        atbat_dict["battedball_cd"] = "F"
        pass
    elif (event_tx == "Lineout" or event_tx == "Line Out" or
          event_tx == "Bunt Lineout"):
        atbat_dict["event_cd"] = 2
        atbat_dict["battedball_cd"] = "L"
        pass
    elif (event_tx == "Pop out" or event_tx == "Pop Out" or
          event_tx == "Bunt Pop Out"):
        atbat_dict["event_cd"] = 2
        atbat_dict["battedball_cd"] = "P"
        pass
    elif (event_tx == "Groundout" or event_tx == "Ground Out" or
          event_tx == "Sac Bunt" or event_tx == "Bunt Groundout"):
        atbat_dict["event_cd"] = 2
        atbat_dict["battedball_cd"] = "G"
        pass

    elif event_tx == "Grounded Into DP":
        atbat_dict["event_cd"] = 2
        atbat_dict["battedball_cd"] = "G"
        pass
    elif event_tx == "Forceout":
        atbat_dict["event_cd"] = 2
        if ab_des.lower().count("grounds") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.lower().count("lines") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.lower().count("flies") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.lower().count("pops") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
        pass
    elif (event_tx == "Double Play" or event_tx == "Triple Play" or
          event_tx == "Sacrifice Bunt D"):
        atbat_dict["event_cd"] = 2
        if ab_des.lower().count("ground") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.lower().count("lines") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.lower().count("flies") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.lower().count("pops") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
        pass
    elif event_tx == "Strikeout" or event_tx == "Strikeout - DP":
        atbat_dict["event_cd"] = 3
        pass
    elif event_tx == "Walk":
        atbat_dict["event_cd"] = 14
        pass
    elif event_tx == "Intent Walk":
        atbat_dict["event_cd"] = 15
        pass
    elif event_tx == "Hit By Pitch":
        atbat_dict["event_cd"] = 16
        pass
    elif event_tx.lower().count("interference") > 0:
        atbat_dict["event_cd"] = 17
        pass
    elif event_tx[-5:] == "Error":
        atbat_dict["event_cd"] = 18
        pass
    elif event_tx == "Fielders Choice Out" or event_tx == "Fielders Choice":
        atbat_dict["event_cd"] = 19
        pass
    elif event_tx == "Single":
        atbat_dict["event_cd"] = 20
        if ab_des.count("on a line drive") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.count("fly ball") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.count("ground ball") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.count("pop up") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
        pass
    elif event_tx == "Double":
        atbat_dict["event_cd"] = 21
        if ab_des.count("line drive") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.count("fly ball") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.count("ground ball") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.count("pop up") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
        pass
    elif event_tx == "Triple":
        atbat_dict["event_cd"] = 22
        if ab_des.count("line drive") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.count("fly ball") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.count("ground ball") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.count("pop up") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
    elif event_tx == "Home Run":
        atbat_dict["event_cd"] = 23
        if ab_des.count("on a line drive") > 0:
            atbat_dict["battedball_cd"] = "L"
            pass
        elif ab_des.count("fly ball") > 0:
            atbat_dict["battedball_cd"] = "F"
            pass
        elif ab_des.count("ground ball") > 0:
            atbat_dict["battedball_cd"] = "G"
            pass
        elif ab_des.count("pop up") > 0:
            atbat_dict["battedball_cd"] = "P"
            pass
        else:
            atbat_dict["battedball_cd"] = ""
            pass
    elif event_tx == "Runner Out":
        if ab_des.lower().count("caught stealing") > 0:
            atbat_dict["event_cd"] = 6
            pass
        elif ab_des.lower().count("picks off") > 0:
            atbat_dict["event_cd"] = 8
            pass
        pass
    else:
        atbat_dict["event_cd"] = 99
        pass
    if ab.find("runner", start="1B"):
        base1 = "1"
        pass
    if ab.find("runner", start="2B"):
        base2 = "2"
        pass
    if ab.find("runner", start="3B"):
        base3 = "3"
        pass

    base_state_tx = base1 + base2 + base3
    atbat_dict["base_start_tx"] = base1 + base2 + base3
    if base_state_tx == "___":
        atbat_dict["start_bases_cd"] = "0"
        pass
    elif base_state_tx == "1__":
        atbat_dict["start_bases_cd"] = "1"
        pass
    elif base_state_tx == "_2_":
        atbat_dict["start_bases_cd"] = "2"
        pass
    elif base_state_tx == "12_":
        atbat_dict["start_bases_cd"] = "3"
        pass
    elif base_state_tx == "__3":
        atbat_dict["start_bases_cd"] = "4"
        pass
    elif base_state_tx == "1_3":
        atbat_dict["start_bases_cd"] = "5"
        pass
    elif base_state_tx == "_23":
        atbat_dict["start_bases_cd"] = "6"
        pass
    elif base_state_tx == "123":
        atbat_dict["start_bases_cd"] = "7"
        pass
    else:
        atbat_dict["start_bases_cd"] = "9"
        pass
    base1 = "_"
    base2 = "_"
    base3 = "_"
    if ab.find("runner", end="1B"):
        base1 = "1"
        pass
    if ab.find("runner", end="2B"):
        base2 = "2"
        pass
    if ab.find("runner", end="3B"):
        base3 = "3"
        pass
    base_state_tx = base1 + base2 + base3
    atbat_dict["base_end_tx"] = base1 + base2 + base3
    if base_state_tx == "___":
        atbat_dict["end_bases_cd"] = "0"
        pass
    elif base_state_tx == "1__":
        atbat_dict["end_bases_cd"] = "1"
        pass
    elif base_state_tx == "_2_":
        atbat_dict["end_bases_cd"] = "2"
        pass
    elif base_state_tx == "12_":
        atbat_dict["end_bases_cd"] = "3"
        pass
    elif base_state_tx == "__3":
        atbat_dict["end_bases_cd"] = "4"
        pass
    elif base_state_tx == "1_3":
        atbat_dict["end_bases_cd"] = "5"
        pass
    elif base_state_tx == "_23":
        atbat_dict["end_bases_cd"] = "6"
        pass
    elif base_state_tx == "123":
        atbat_dict["end_bases_cd"] = "7"
        pass
    else:
        atbat_dict["end_bases_cd"] = "9"
        pass

    if debug > 1:
        print("Parsed atbat", flush=True)
        pass
    return atbat_dict


def parse_pitch(pitch, event_cd, strike_tally, ball_tally):
    pitch_dict = {}

    pitch_dict["pa_terminal_fl"] = "U"
    if 'type' in pitch.attrs:
        pitch_res = pitch["type"]
        pass
    else:
        pitch_res = ""
        pass
    if 'des' in pitch.attrs:
        pitch_des = pitch["des"]
        pass
    else:
        pitch_des = ""
        pass
    pitch_dict["pitch_des"] = pitch_des
    if pitch_des == "Foul":
        pitch_res = "F"
        pass
    if pitch_des == "Called Strike":
        pitch_res = "C"
        pass
    pitch_dict["pitch_res"] = pitch_res

    if 'id' in pitch.attrs:
        pitch_dict["pitch_id"] = pitch["id"]
        pass
    else:
        pitch_dict["pitch_id"] = ""
        pass

    if (pitch_res == "X" or
        ((pitch_res == "S" or pitch_res == "C") and event_cd == 3 and
         strike_tally == 2) or
        (ball_tally == 3 and pitch_res == "B" and
         (event_cd == 14 or event_cd == 15))):
        pitch_dict["pa_terminal_fl"] = "T"
        pass
    else:
        pitch_dict["pa_terminal_fl"] = "F"
        pass
    if 'x' in pitch.attrs:
        pitch_dict["x"] = pitch["x"]
        pass
    else:
        pitch_dict["x"] = ""
        pass
    if 'y' in pitch.attrs:
        pitch_dict["pitch_y"] = pitch["y"]
        pass
    else:
        pitch_dict["pitch_y"] = ""
        pass
    if 'sv_id' in pitch.attrs:
        pitch_dict["sv_id"] = pitch["sv_id"]
        pass
    else:
        pitch_dict["sv_id"] = ""
        pass
    if 'start_speed' in pitch.attrs:
        pitch_dict["start_speed"] = pitch["start_speed"]
        pass
    else:
        pitch_dict["start_speed"] = ""
        pass
    if 'end_speed' in pitch.attrs:
        pitch_dict["end_speed"] = pitch["end_speed"]
        pass
    else:
        pitch_dict["end_speed"] = ""
        pass
    if 'sz_top' in pitch.attrs:
        pitch_dict["sz_top"] = pitch["sz_top"]
        pass
    else:
        pitch_dict["sz_top"] = ""
        pass
    if 'sz_bot' in pitch.attrs:
        pitch_dict["sz_bot"] = pitch["sz_bot"]
        pass
    else:
        pitch_dict["sz_bot"] = ""
        pass
    if 'pfx_x' in pitch.attrs:
        pitch_dict["pfx_x"] = pitch["pfx_x"]
        pass
    else:
        pitch_dict["pfx_x"] = ""
        pass
    if 'pfx_z' in pitch.attrs:
        pitch_dict["pfx_z"] = pitch["pfx_z"]
        pass
    else:
        pitch_dict["pfx_z"] = ""
        pass
    if 'px' in pitch.attrs:
        pitch_dict["px"] = pitch["px"]
        pass
    else:
        pitch_dict["px"] = ""
        pass
    if 'pz' in pitch.attrs:
        pitch_dict["pz"] = pitch["pz"]
        pass
    else:
        pitch_dict["pz"] = ""
        pass
    if 'x0' in pitch.attrs:
        pitch_dict["x0"] = pitch["x0"]
        pass
    else:
        pitch_dict["x0"] = ""
        pass
    if 'y0' in pitch.attrs:
        pitch_dict["y0"] = pitch["y0"]
        pass
    else:
        pitch_dict["y0"] = ""
        pass
    if 'z0' in pitch.attrs:
        pitch_dict["z0"] = pitch["z0"]
        pass
    else:
        pitch_dict["z0"] = ""
        pass
    if 'vx0' in pitch.attrs:
        pitch_dict["vx0"] = pitch["vx0"]
        pass
    else:
        pitch_dict["vx0"] = ""
        pass
    if 'vy0' in pitch.attrs:
        pitch_dict["vy0"] = pitch["vy0"]
        pass
    else:
        pitch_dict["vy0"] = ""
        pass
    if 'vz0' in pitch.attrs:
        pitch_dict["vz0"] = pitch["vz0"]
        pass
    else:
        pitch_dict["vz0"] = ""
        pass
    if 'ax' in pitch.attrs:
        pitch_dict["ax"] = pitch["ax"]
        pass
    else:
        pitch_dict["ax"] = ""
        pass
    if 'ay' in pitch.attrs:
        pitch_dict["ay"] = pitch["ay"]
        pass
    else:
        pitch_dict["ay"] = ""
        pass
    if 'az' in pitch.attrs:
        pitch_dict["az"] = pitch["az"]
        pass
    else:
        pitch_dict["az"] = ""
        pass
    if 'break_y' in pitch.attrs:
        pitch_dict["break_y"] = pitch["break_y"]
        pass
    else:
        pitch_dict["break_y"] = ""
        pass
    if 'break_angle' in pitch.attrs:
        pitch_dict["break_angle"] = pitch["break_angle"]
        pass
    else:
        pitch_dict["break_angle"] = ""
        pass
    if 'break_length' in pitch.attrs:
        pitch_dict["break_length"] = pitch["break_length"]
        pass
    else:
        pitch_dict["break_length"] = ""
        pass
    if 'pitch_type' in pitch.attrs:
        pitch_dict["pitch_type"] = pitch["pitch_type"]
        pass
    else:
        pitch_dict["pitch_type"] = ""
        pass
    if 'type_confidence' in pitch.attrs:
        pitch_dict["type_conf"] = pitch["type_confidence"]
        pass
    else:
        pitch_dict["type_conf"] = ""
        pass
    if 'zone' in pitch.attrs:
        pitch_dict["zone"] = pitch["zone"]
        pass
    else:
        pitch_dict["zone"] = ""
        pass
    if 'spin_dir' in pitch.attrs:
        pitch_dict["spin_dir"] = pitch["spin_dir"]
        pass
    else:
        pitch_dict["spin_dir"] = ""
        pass
    if 'spin_rate' in pitch.attrs:
        pitch_dict["spin_rate"] = pitch["spin_rate"]
        pass
    else:
        pitch_dict["spin_rate"] = ""
        pass
    if 'nasty' in pitch.attrs:
        pitch_dict["nasty"] = pitch["nasty"]
        pass
    else:
        pitch_dict["nasty"] = ""
        pass

    return pitch_dict


def parse_inning_action(item, torb):
    global atbats, pitches
    if debug > 1:
        print("In Action", flush=True)
        pass
    if item.name == "atbat":
        if debug > 1:
            print("Parse atbat", flush=True)
            pass
        atbat_dict = parse_atbat(item)

        [batter_name, batter_side,
         batter_id,
         pitcher_name, pitcher_hand,
         pitcher_id] = [atbat_dict["bat_name"],
                        atbat_dict["bat_hand_cd"],
                        atbat_dict["bat_mlbid"],
                        atbat_dict["pit_name"],
                        atbat_dict["pit_hand_cd"],
                        atbat_dict["pit_mlbid"]]
        # print("Atbat {}: {}-{} vs {}-{}".format(item["num"], batter_name, batter_side, pitcher_name, pitcher_hand))
        pitch_seq = ""
        pitch_type_seq = ""
        ball_tally = 0
        strike_tally = 0
        if debug > 1:
            print("Parsing pitches", flush=True)
            pass
        for i in item.children:
            if debug > 1:
                print("Parsing child", flush=True)
                pass
            if i.name == "pitch":
                if debug > 1:
                    print("Parsing pitch", flush=True)
                    pass
                pitch_dict = parse_pitch(i, atbat_dict["event_cd"], strike_tally, ball_tally)
                if debug > 1:
                    print("Parse pitch", flush=True)
                    pass
                # Add to dictionary!!!
                pitch_seq += pitch_dict["pitch_res"]
                pitch_dict["pitch_seq"] = pitch_seq
                atbat_dict["pitch_seq"] = pitch_seq

                pitch_type_seq += "|" + pitch_dict["pitch_type"]
                pitch_dict["pitch_type_seq"] = pitch_type_seq
                atbat_dict["pitch_type_seq"] = pitch_type_seq

                if pitch_dict["pitch_res"] == "B" and ball_tally < 4:
                    ball_tally += 1
                    pass
                pitch_dict["ball_tally"] = ball_tally
                atbat_dict["ball_tally"] = ball_tally

                if ((pitch_dict["pitch_res"] == "S" or
                     pitch_dict["pitch_res"] == "C") and strike_tally < 3):
                    strike_tally += 1
                    pass
                pitch_dict["strike_tally"] = strike_tally
                atbat_dict["strike_tally"] = strike_tally

                pitch_dict["batter_name"] = batter_name
                pitch_dict["batter_side"] = batter_side
                pitch_dict["batter_id"] = batter_id
                pitch_dict["pitcher_name"] = pitcher_name
                pitch_dict["pitcher_hand"] = pitcher_hand
                pitch_dict["pitcher_id"] = pitcher_id

                for k, v in game_dict.items():
                    if (k == "curr_away_catcher_id" or
                        k == "curr_home_catcher_id" or
                        k == "curr_away_catcher_name" or
                        k == "curr_home_catcher_name"):
                        pass
                    else:
                        pitch_dict[k] = v
                        atbat_dict[k] = v
                        pass
                    pass

                if torb == "top":
                    pitch_dict["catcher_name"] = game_dict["curr_home_catcher_name"]
                    pitch_dict["catcher_id"] = game_dict["curr_home_catcher_id"]
                    pass
                else:
                    pitch_dict["catcher_name"] = game_dict["curr_away_catcher_name"]
                    pitch_dict["catcher_id"] = game_dict["curr_away_catcher_id"]
                    pass

                pitches.append(pitch_dict)
                pass
            elif i.name == "action":
                parse_action(i, torb, 1)
                pass
            elif i.name == "runner":
                pass
            pass
        atbats.append(atbat_dict)
        if debug > 1:
            print("Parsed atbat", flush=True)
            pass
        pass
    elif item.name == "action":
        if debug > 1:
            print("Parse action", flush=True)
            pass
        parse_action(item, torb, 0)
        if debug > 1:
            print("Parsed action", flush=True)
            pass
        pass
    return


def parse_inning(url):
    while True:
        try:
            xmlfile = BeautifulSoup(urlopen(url), "xml")
            break
        except urllib.error.URLError:
            time.sleep(10)
            pass
        pass
    if debug > 0:
        print("Parsed inning", flush=True)
        pass
    inning_number = xmlfile.inning["num"]
    # print(inning_number)
    if xmlfile.inning.find("top"):
        if debug > 1:
            print("Top", flush=True)
            pass
        for item in xmlfile.inning.top.children:
            parse_inning_action(item, 'top')
        pass
    if xmlfile.inning.find("bottom"):
        if debug > 1:
            print("Bottom", flush=True)
            pass
        for item in xmlfile.inning.bottom.children:
            parse_inning_action(item, 'bottom')
        pass
    return


def read_innings(inn_url):
    for inning in BeautifulSoup(urlopen(inn_url), "xml").find_all("a", href=re.compile("inning_\d*.xml")):
        if debug > 0:
            print("inning {}".format(inning.get_text().strip()))
            pass
        parse_inning(inn_url + inning.get_text().strip())
        pass
    return


def read_game(g_url, date):
    # Setup initial values
    game_dict['st_fl'] = "U"
    game_dict['regseason_fl'] = "U"
    game_dict['playoff_fl'] = "U"
    game_dict['game_type'] = "U"
    game_dict['game_type_des'] = "Unknown"
    game_dict['local_game_time'] = "Unknown"
    game_dict['game_id'] = "Unknown"
    game_dict['home_team_id'] = "Unknown"
    game_dict['away_team_id'] = "Unknown"
    game_dict['home_team_lg'] = "Unknown"
    game_dict['away_team_lg'] = "Unknown"
    game_dict['interleague_fl'] = "U"
    game_dict['park_id'] = "Unknown"
    game_dict['park_name'] = "Unknown"
    game_dict['park_loc'] = "Unknown"
    game_dict["curr_home_catcher_id"] = "Unknown"
    game_dict["curr_home_catcher_name"] = "Unknown"
    game_dict["curr_away_catcher_id"] = "Unknown"
    game_dict["curr_away_catcher_name"] = "Unknown"
    game_dict["hp_umpire_id"] = "Unknown"
    game_dict["hp_umpire_name"] = "Unknown"
    game_dict["venue_name"] = "Unknown"
    game_dict["start_time"] = "Unknown"
    game_dict["weather"] = "Unknown"
    game_dict["wind"] = "Unknown"
    game_dict["umpire"] = "Unknown"
    game_dict["date"] = date

    # Read in the local game file
    if BeautifulSoup(urlopen(g_url), "xml").find("a", href="game.xml"):
        url = g_url + "game.xml"
        if debug > 0:
            print("Parse game: {}".format(url))
            pass
        parse_game(url)
        pass

    # Read in boxscore to get all catchers that played
    if BeautifulSoup(urlopen(g_url), "xml").find("a", href="boxscore.xml"):
        if debug > 0:
            print("Parse game: {}".format(url))
            pass
        url = g_url + "boxscore.xml"
        parse_boxscore(url)
        pass

    # Read in raw boxscore to get umpire and venue and game information
    if BeautifulSoup(urlopen(g_url), "xml").find("a", href="rawboxscore.xml"):
        if debug > 0:
            print("Parse game: {}".format(url))
            pass
        url = g_url + "rawboxscore.xml"
        parse_rawBoxscore(url)
        pass

    # Read in players to get home plate umpire
    if BeautifulSoup(urlopen(g_url), "xml").find("a", href="players.xml"):
        if debug > 0:
            print("Parse game: {}".format(url))
            pass
        url = g_url + "players.xml"
        parse_players(url)
        pass

    # Begin walking through the innings to grab all of the events.
    # inn_url = g_url + "inning/"
    # try:
    #     urlopen(inn_url)
    #     tested_inn_url = inn_url
    #     read_innings(tested_inn_url)
    # except:
    #     print("WARN: Couldn't find inning directory at {}".format(inn_url))
    #     pass

    if BeautifulSoup(urlopen(g_url), "xml").find("a", href="inning/"):
        inn_url = g_url + "inning/"
        read_innings(inn_url)
        pass
    else:
        print("WARN: Couldn't find inning directory at {}".format(g_url))
        pass

    if debug > 5:
        game_keys = []
        for k in game_dict:
            game_keys.append(k)
            pass
        with open("game.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=game_keys)
            writer.writeheader()
            writer.writerow(game_dict)
            pass
        pass

    return


def read_yearmonth(year, month, day=0):
    global atbats, pitches
    atbats = []
    pitches = []

    endyear = year
    endmonth = month + 1
    if endmonth > 12:
        endyear += 1
        endmonth = 1
        pass

    startdate = datetime.date(year, month, 1)
    enddate = datetime.date(endyear, endmonth, 1)

    # if a day is specified, just get that one day
    if day:
        startdate = datetime.date(year, month, day)
        enddate = startdate + datetime.timedelta(days=1)
    delta = enddate - startdate

    for i in range(delta.days):
        active_date = startdate + datetime.timedelta(days=i)
        print(active_date)
        y = str(active_date.year)
        m = active_date.strftime('%m')
        d = active_date.strftime('%d')
        datestr = y + "-" + m + "-" + d
        # url = base_url + "year_" + y + "/month_" + m + "/day_" + d + "/"
        url = base_url + "year_" + y + "/month_" + m + "/day_" + d
        # print(url)

        good_url = 0
        try:
            urlopen(url)
            good_url = 1
            pass
        except:
            print("ERR: {} couldn't be opened".format(url))
            good_url = 0
            pass

        if good_url:
            day_dir = BeautifulSoup(urlopen(url), "xml")
            for game in day_dir.find_all("a", href=re.compile("gid_.*")):
                g = game.get_text().strip()
                if g[len(g) - 2: len(g) - 1] == type(int(1)):
                    game_number = g[len(g) - 2: len(g) - 1]
                    pass
                else:
                    game_number = 1
                    pass
                time.sleep(1)
                g_url = url + "/" + g
                print(g_url)
                good1 = 0
                try:
                    urlopen(g_url)
                    good1 = 1
                    pass
                except:
                    print("ERR: Game not found")
                    pass
                if good1:
                    read_game(g_url, datestr)
                    pass
                pass
            pass
        pass

    # Output what we have found
    # for k, v in game_dict.items():
    #    print("{} => {}".format(k, v))
    #    pass

    if day:
        day_str = "-" + str(day)
    else:
        day_str = ""

    atbat_file = "atbats/atbats_" + str(year) + "-" + str(month) + day_str + ".csv"
    pitches_file = "pitches/pitches_" + str(year) + "-" + str(month) + day_str + ".csv"

    if len(atbats) > 0:
        atbat_keys = []
        for k in sorted(atbats[0]):
            atbat_keys.append(k)
            pass
        with open(atbat_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=atbat_keys)
            writer.writeheader()
            for dict in atbats:
                writer.writerow(dict)
                pass
            pass
        pass

    if len(pitches) > 0:
        pitch_keys = []
        for k in sorted(pitches[0]):
            pitch_keys.append(k)
            pass
        with open(pitches_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=pitch_keys)
            writer.writeheader()
            for dict in pitches:
                writer.writerow(dict)
                pass
            pass
            return pitches
        pass

    return


def read_year(year):
    if test == 1:
#        g_url = "http://gd2.mlb.com/components/game/mlb/year_2015/month_04/day_07/gid_2015_04_07_sfnmlb_arimlb_1/"
#        read_game(g_url)
        g_url = "http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_03/gid_2015_03_03_bocbbc_bosmlb_2/"
        read_game(g_url, '2015-03-03')
        g_url = "http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_02/gid_2015_03_02_flsbbc_detmlb_1/"
        read_game(g_url, '2015-03-03')
        g_url = "http://gd2.mlb.com/components/game/mlb/year_2015/month_03/day_03/gid_2015_03_03_asubbc_arimlb_1/"
        read_game(g_url, '2015-03-03')
        g_url = "http://gd2.mlb.com/components/game/mlb/year_2015/month_06/day_30/gid_2015_06_30_pitmlb_detmlb_1/"
        read_game(g_url, '2015-03-03')

        return
    elif test == 2:
        read_yearmonth(year, 3)
        return

    for month in range(1, 13):
        # print("Month: {}".format(month))
        read_yearmonth(year, month)

    return


def main():
    global debug, test

    parser = ap.ArgumentParser()
    parser.add_argument("year", type=int,
                        help="year to pull at bats and pitches from")
    parser.add_argument("-m", "--month", type=int,
                        help="month of year to pull at bats and pitches from")
    parser.add_argument("-d", "--day", type=int,
                        help="day of month to pull at bats and pitches from")
    parser.add_argument("--debug", type=int,
                        help="debug flag")
    parser.add_argument("-t", "--test", type=int,
                        help="test flag")

    args = parser.parse_args()

    if args.debug:
        debug = args.debug
        pass

    if args.test:
        test = args.test
        read_year(2015)
        return

    if args.year:
        if args.month:
            read_yearmonth(args.year, args.month)
            pass
        else:
            read_year(args.year)
            pass
        pass
    else:
        print("ERR: No year specified")
        pass

    if args.day:
        if args.month:
            read_yearmonth(args.year, args.month, args.day)
            pass
        else:
            print("ERR: No month specified for the day of the month")
            pass
        pass

    return


if __name__ == '__main__':
    main()
