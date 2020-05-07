import 'package:flutter/material.dart';
import 'package:innebandy_test1/playerPage.dart';
import 'package:innebandy_test1/readFile.dart';
import 'package:innebandy_test1/teamPage.dart';
import 'LocalDatabase.dart';

class SearchPage extends StatefulWidget {
  @override
  _SearchPageState createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage> {
  final TextEditingController _filter = new TextEditingController();
  String _searchText = "";

  List<dynamic> teams = LocalDatabase.getAllTeamsAndPlayers();
  List<dynamic> filteredTeams = LocalDatabase.getAllTeamsAndPlayers();
  Icon _searchIcon = new Icon(Icons.search);
  Widget _appBarTitle = new Text("Klicka på ikonen för att söka");

  _SearchPageState() {
    _filter.addListener(() {
      if (_filter.text.isEmpty) {
        setState(() {
          _searchText = "";
          filteredTeams = teams;
        });
      } else {
        setState(() {
          _searchText = _filter.text;
        });
      }
    });
  }

  @override
  void initState() {
    super.initState();
  }

  void _searchPressed() {
    setState(() {
      if (this._searchIcon.icon == Icons.search) {
        this._searchIcon = new Icon(Icons.close);
        this._appBarTitle = new TextField(
          controller: _filter,
          decoration: new InputDecoration(
              prefixIcon: new Icon(Icons.search), hintText: 'Search...'),
        );
      } else {
        this._searchIcon = new Icon(Icons.search);
        this._appBarTitle = new Text('Search Example');
        filteredTeams = teams;
        _filter.clear();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildBar(context),
      body: Container(child: _buildList()),
      resizeToAvoidBottomPadding: false,
    );
  }

  Widget _buildBar(BuildContext context) {
    return new AppBar(
      centerTitle: true,
      title: _appBarTitle,
      leading: new IconButton(
        icon: _searchIcon,
        onPressed: _searchPressed,
      ),
      actions: <Widget>[],
    );
  }

  Widget _buildList() {
    if ((_searchText.isNotEmpty)) {
      List<dynamic> tempList = new List();
      for (int i = 0; i < filteredTeams.length; i++) {
        if (filteredTeams.elementAt(i) is Teams) {
          if (filteredTeams
              .elementAt(i)
              .teamName
              .toLowerCase()
              .contains(_searchText.toLowerCase())) {
            tempList.add(filteredTeams[i]);
          }
        } else {
          if (filteredTeams
              .elementAt(i)
              .name
              .toLowerCase()
              .contains(_searchText.toLowerCase())) {
            tempList.add(filteredTeams[i]);
          }
        }
      }

      filteredTeams = tempList;
    }
    return ListView.builder(
      itemCount: teams == null ? 0 : filteredTeams.length,
      itemBuilder: (BuildContext context, int index) {
        if (filteredTeams.elementAt(index) is Teams) {
          return teamTile(filteredTeams.elementAt(index), context);
        } else {
          return playerTile(filteredTeams.elementAt(index), context);
        }
      },
    );
  }
}
