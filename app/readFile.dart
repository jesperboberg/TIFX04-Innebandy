import 'dart:convert';
import 'package:http/http.dart' as http;

Future<List<Teams>> getTheFile() async {
    //var url= 'http://hindret.eu/IB/jonteinput.json'; // for collecting data live
    var url='https://raw.githubusercontent.com/jesperboberg/TIFX04-Innebandy/master/app/input.json';
    var response = await http.get(url);
     List<Teams> teams = new List();
  List<Player> teamA = new List();
  List<Player> teamB = new List();
  List<String> teamNames = new List();
     if (response.statusCode == 200) {
      Map<String,dynamic> notesJson = json.decode(response.body);
        for (var i = 0; i <notesJson['teams'].length; i++) {
    teamNames.add(notesJson['teams'][i]['teamName']);
    for (var j = 0; j < notesJson['teams'][i]['players'].length; j++) {
      if (i == 0) {
        teamA.add(new Player(
            notesJson['teams'][i]['players'][j]['name'],
            notesJson['teams'][i]['players'][j]['number'],
            notesJson['teams'][i]['players'][j]['distance'],
            notesJson['teams'][i]['players'][j]['standPart'],
            notesJson['teams'][i]['players'][j]['walkPart'],
            notesJson['teams'][i]['players'][j]['jogPart'],
            notesJson['teams'][i]['players'][j]['runPart'],
            notesJson['teams'][i]['teamName']));
      } else {
        teamB.add(new Player(
            notesJson['teams'][i]['players'][j]['name'],
            notesJson['teams'][i]['players'][j]['number'],
            notesJson['teams'][i]['players'][j]['distance'],
            notesJson['teams'][i]['players'][j]['standPart'],
            notesJson['teams'][i]['players'][j]['walkPart'],
            notesJson['teams'][i]['players'][j]['jogPart'],
            notesJson['teams'][i]['players'][j]['runPart'],
            notesJson['teams'][i]['teamName']));
      }
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
  String team;
  int standPart;
  int walkPart;
  int jogPart;
  int runPart;
  Player(this.name, this.nr, this.distance, this.standPart, this.walkPart, this.jogPart, this.runPart, this.team);
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
