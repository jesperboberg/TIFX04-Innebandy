import 'package:innebandy_test1/readFile.dart';
class LocalDatabase {
static List<Teams> _teams;

static Future<bool> fetchTeams() async {
  _teams = await getTheFile();
  return true;
}

static getAllTeams(){
  return _teams;
}
static getAllTeamsAndPlayers(){
  List<dynamic> teamsAndPlayers = new List();
  for (Teams team in _teams){
   teamsAndPlayers.add(team); 
  }
  for (Teams team in _teams){
   for (Player pl in team.players){
    teamsAndPlayers.add(pl);
   }
  }
  return teamsAndPlayers;
}
static getAllPlayers(){
  List<dynamic> allPlayers = new List();
  for (Teams team in _teams){
   for (Player pl in team.players){
    allPlayers.add(pl);
   }
  }
  return allPlayers;
}
}
