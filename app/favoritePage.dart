import 'package:flutter/material.dart';
import 'package:innebandy_test1/LocalDatabase.dart';
import 'package:innebandy_test1/playerPage.dart';
import 'drawer.dart';
import 'readFile.dart';
import 'teamPage.dart';


Widget favoritePage(BuildContext context) {
  List<dynamic> teamsandplayers = LocalDatabase.getAllTeamsAndPlayers();
  List<dynamic> tempList = new List<dynamic>();
  Future<dynamic> _future = SharedPreferenceHelper.getFavoritesSF();
  
  return Scaffold(
    appBar: AppBar(
                title: Text('Mina favoriter'),
                actions: <Widget>[
                  GestureDetector(
                    onTap: () {
                      SharedPreferenceHelper.resetSF();
                    },
                    child: Icon(Icons.restore),
                  ),
                 
                ],
              ),
    body: FutureBuilder<dynamic>(
        future: _future,
        builder: (BuildContext context, AsyncSnapshot<dynamic> snapshot) {
          if (snapshot.data == true) {
            for (String str in SharedPreferenceHelper.favoriteList) {
              for (int i = 0; i < teamsandplayers.length; i++) {
                if (teamsandplayers.elementAt(i) is Teams) {
                  if (teamsandplayers
                      .elementAt(i)
                      .teamName
                      .toLowerCase()
                      .contains(str.toLowerCase())) {
                    tempList.add(teamsandplayers[i]);
                  }
                } else {
                  if (teamsandplayers
                      .elementAt(i)
                      .name
                      .toLowerCase()
                      .contains(str.toLowerCase())) {
                    tempList.add(teamsandplayers[i]);
                  }
                }
              }
            }
            
            return ListView.builder(
              itemCount: SharedPreferenceHelper.favoriteList.length,
              itemBuilder: (BuildContext context, int index) {
                if (tempList.elementAt(index) is Teams) {
                  return teamTile(tempList.elementAt(index), context);
                } else {
      
                  return playerTile(tempList.elementAt(index), context);
                }
              },
            );
          } else {
            return CircularProgressIndicator();
          }
        }),
        drawer: drawer(context),
  );
}
