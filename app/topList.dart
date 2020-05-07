import 'package:flutter/material.dart';
import 'package:grafpix/icons.dart';
import 'package:innebandy_test1/LocalDatabase.dart';
import 'package:grafpix/pixbuttons/medal.dart';
import 'package:innebandy_test1/drawer.dart';
import 'readFile.dart';

Widget topList(BuildContext context) {
  List<dynamic> allPlayers = LocalDatabase.getAllPlayers();
  List<Widget> tiles = new List();
  allPlayers.sort((a, b) => b.distance.compareTo(a.distance));
  int position = 0;
  for (Player pl in allPlayers) {
    position = position + 1;
    tiles.add(topListss(pl, context, position));
  }
  return Scaffold(
  appBar: AppBar(
                title: Text('Topplista'),
               
              ),
  body: ListView(children: tiles),
  drawer: drawer(context));
}

Widget topListss(Player player, context, int position) {
  PixMedal _leading;
  if (position == 1) {
    _leading = PixMedal(
      icon: PixIcon.shopware,
      medalType: MedalType.Gold,
      radius: 18.0,
      iconSize: 30.0,
    );
  }
  if (position == 2) {
    _leading = PixMedal(
      icon: PixIcon.shopware,
      medalType: MedalType.Silver,
      radius: 18.0,
      iconSize: 30.0,
    );
  }
  if (position == 3) {
    _leading = PixMedal(
      icon: PixIcon.shopware,
      medalType: MedalType.Bronze,
      radius: 18.0,
      iconSize: 30.0,
    );
  }
  if (position > 3) {
  }
  return ListTile(
    title: Text(
        '$position    ' +
            player.name +
            ' har sprungit ' +
            '${player.distance}' +
            ' meter',
        style: TextStyle(
          fontWeight: FontWeight.w500,
          fontSize: 15,
        )),
    trailing: _leading,
  );
}
