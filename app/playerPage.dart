import 'package:flutter/material.dart';
import 'package:innebandy_test1/readFile.dart';

Widget playerPage2(BuildContext context, List<Player> players){
  List<Widget> tiles = new List();
  for (Player pl in players){
    tiles.add(_tile(pl, context));
  }
  return ListView(
    children: tiles);
      
}

Widget _tile(Player player, context) => Card(
      elevation: 5,
      child: ListTile(
        title: Text(player.name,
            style: TextStyle(
              fontWeight: FontWeight.w500,
              fontSize: 20,
            )),
        subtitle: Text(' number: ${player.nr}'),
        leading: Icon(Icons.star),
        trailing: Icon(Icons.keyboard_arrow_right),
        onTap: () {
          Navigator.push(context,
              new MaterialPageRoute(builder: (ctxt) => new StatsPage(player)));
        },
      ),
    );

class StatsPage extends StatelessWidget {
  final Player pl;
  StatsPage(this.pl);

  @override
  Widget build(BuildContext ctxt) {
    return new Scaffold(
      appBar: new AppBar(
        title: new Text("Statistik för ${pl.name}"),
      ),
      body:
          Center(child: new Text(" ${pl.name} has run ${pl.distance} meters.")),
    );
  }
}
