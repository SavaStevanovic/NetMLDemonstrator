import json
import os 

def getUniversalNames(file):
        with open(file, encoding='utf-8', mode = 'r') as fp:
            universalNames = json.load(fp)
        return universalNames

def HelperMethod(un, argString, bookingName):
    args = argString.split(',')
    if len(args) == 1:
        return args[0]
    result = un[args[0]]
    for name in args[1:]:
        if name == args[-1] and name not in result:
            message = '(getNamingDictFromFile method) Club with name: {0} not in booking_names: {1}, with liga path: {2}'.format(name, bookingName, argString)
            return None

    result = result[name]
    return result

def getNamingDictFromFile(bookingNameFile, un, mapping_dict, league_dict):
    with open(bookingNameFile, encoding='utf-8', mode = 'r') as fp:
        bookingNameDjson = fp.read()
        bookingNameD = json.loads(bookingNameDjson)
        for sportname in bookingNameD:
            if sportname == 'name':
                bookingNameD[sportname] = HelperMethod(un, bookingNameD[sportname], bookingNameFile)
                continue
            for liganame in bookingNameD[sportname]:
                if liganame == 'name':
                    bookingNameD[sportname]['name'] = HelperMethod(un, bookingNameD[sportname]['name'], bookingNameFile)
                    continue
                for teamname in bookingNameD[sportname][liganame]:
                    if teamname.startswith('__') or teamname=='' or teamname=='name':
                        continue
                    # bookingNameD[sportname][liganame][teamname] = HelperMethod(un, bookingNameD[sportname][liganame][teamname], bookingName)
                    key = bookingNameD[sportname][liganame][teamname]
                    if key not in mapping_dict:
                        mapping_dict[key] = []
                    valid_team_name = convert_to_valid_string(teamname)
                    mapping_dict[key] += [valid_team_name]
                    league_key = ','.join(key.split(',')[:-1])
                    if league_key not in league_dict:
                        league_dict[league_key] = []
                    league_dict[league_key] += [key.split(',')[-1]]
        return mapping_dict

def convert_to_valid_string(orig_string):
    return ''.join(convert_to_valid_char(x) for x in orig_string)

def convert_to_valid_char(x):
    ord_x = ord(x)
    if (ord_x<=122 and ord_x>=97) or x in [' ', '.']:
        return x
    return '*'