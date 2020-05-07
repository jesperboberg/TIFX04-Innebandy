import 'package:flutter/material.dart';
import 'package:innebandy_test1/playerPage.dart';
import 'package:innebandy_test1/readFile.dart';

Widget teamTile(Teams teams, context) => Card(
  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30.0),),
      elevation: 5,
      child: ListTile(
        title: Text(teams.teamName,
            style: TextStyle(
              fontWeight: FontWeight.w500,
              fontSize: 20,
            )),
        leading: Icon(IconData(59375, fontFamily: 'MaterialIcons')),
        trailing: Icon(Icons.keyboard_arrow_right),
        onTap: () {
          Navigator.push(
              context,
              new MaterialPageRoute(
        builder: (ctxt) => new PlayerPage(teams.teamName, teams.players)));
        },
      ),
    );
class PlayerPage extends StatelessWidget {
  final List<Player> pl;
  final String teamname;
  PlayerPage(this.teamname, this.pl);

  @override
  Widget build(BuildContext ctxt) {
    
    return new Scaffold(
      appBar: new AppBar(
        title: new Text(teamname),
      ),
      body: 
          Center(child: playerPage2(ctxt, pl, teamname)),
    );
  }
}
