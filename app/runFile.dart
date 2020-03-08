//import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:flutter/services.dart';





Future<List<Teams>> getTheFile() async {
  String fileText = await rootBundle.loadString('assets/text/input.json');
  Map<String, dynamic> input = jsonDecode(fileText);
  List<Teams> teams = new List();
  List<Player> teamA = new List();
  List<Player> teamB = new List();
  List<String> teamNames = new List();
  for (var i = 0; i < input['teams'].length; i++) {
    teamNames.add(input['teams'][i]['teamName']);
    for (var j = 0; j < input['teams'][i]['players'].length; j++) {
      if (i == 0) {
        teamA.add(new Player(
            input['teams'][i]['players'][j]['name'],
            input['teams'][i]['players'][j]['number'],
            input['teams'][i]['players'][j]['distance']));
      } else {
        teamB.add(new Player(
            input['teams'][i]['players'][j]['name'],
            input['teams'][i]['players'][j]['number'],
            input['teams'][i]['players'][j]['distance']));
      }
    }
  }

  teams.add(new Teams(teamNames.elementAt(0), teamA));
  teams.add(new Teams(teamNames.elementAt(1), teamB));
  return teams;
}

class Player {
  String name;
  int nr;
  int distance;
  Player(this.name, this.nr, this.distance);
}

class Teams {
  String teamName;
  List<Player> players;
  Teams(this.teamName, this.players);
}

class TeamsList {
  List<Teams> teams;
  TeamsList(this.teams);
}
