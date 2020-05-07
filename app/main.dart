import 'package:flutter/material.dart';
import 'package:innebandy_test1/LocalDatabase.dart';
import 'package:innebandy_test1/SearchPage.dart';
import 'package:innebandy_test1/webScraper.dart';
//import 'package:shared_preferences/shared_preferences.dart';
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'favoritePage.dart';
import 'homePage.dart';
import 'themeChanger.dart';
import 'topList.dart';
//import 'playerPage.dart'; uncomment line 69 first time running the app

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  static const String _title = 'Flutter Code Sample';
  @override
  Widget build(BuildContext context) {
    return ThemeBuilder(
        defaultBrightness: Brightness.dark,
        builder: (context, _brightness) {
          return MaterialApp(
            title: _title,
            theme: ThemeData(
              primarySwatch: Colors.blue,
              brightness: _brightness,
              fontFamily: 'Georgia',
              appBarTheme: AppBarTheme(
                color: Colors.black,
                textTheme: TextTheme(
                  headline:
                      TextStyle(fontSize: 15.0, fontWeight: FontWeight.bold),
                  title: TextStyle(fontSize: 15.0, fontStyle: FontStyle.italic),
                  body1: TextStyle(fontSize: 14.0, fontFamily: 'Hind'),
                ),
              ),
            ),
            home: MyStatefulWidget(),
          );
        });
  }
}

class MyStatefulWidget extends StatefulWidget {
  MyStatefulWidget({Key key}) : super(key: key);

  @override
  _MyStatefulWidgetState createState() => _MyStatefulWidgetState();
}

class _MyStatefulWidgetState extends State<MyStatefulWidget> {
  int _selectedIndex = 0;
  static final showGrid = true;

 

  static const TextStyle optionStyle =
      TextStyle(fontSize: 30, fontWeight: FontWeight.bold);

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    Future<bool> bo = SSLResponse.fetchTeams();
//Future<bool> boo= SharedPreferenceHelper.addToFavoritesSF('Artur Engstrom');
    return FutureBuilder(
        future: LocalDatabase.fetchTeams(),
        builder: (context, snapshot) {
          if (snapshot != null && snapshot.hasData) {
            final List<Widget> _children = [
              Home(),
              SearchPage(),
              favoritePage(context),
              topList(context),
            ];
            return Scaffold(
              
              body: _children.elementAt(_selectedIndex),
              bottomNavigationBar: CurvedNavigationBar(
                backgroundColor: Colors.white,
                color: Colors.grey,
                items: <Widget>[
                  Icon(Icons.home, size: 20),
                  Icon(Icons.search, size: 20),
                  Icon(Icons.favorite, size: 20),
                  Icon(IconData(59375, fontFamily: 'MaterialIcons'), size: 20),
                  
                ],
                onTap: _onItemTapped,
              ),
            );
          } else {
            return Container(child: Center(child: CircularProgressIndicator()));
          }
        });
  }
}

