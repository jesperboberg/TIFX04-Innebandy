import 'package:flutter/material.dart';
import 'package:innebandy_test1/teamPage.dart';
import 'LivePage.dart';
import 'playerPage.dart';
import 'drawer.dart';

void main() => runApp(MyApp());

/// This Widget is the main application widget.
class MyApp extends StatelessWidget {
  static const String _title = 'Flutter Code Sample';
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: _title,
      home: MyStatefulWidget(),
    );
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
    List<String> _pageTitle = ['Live', 'Teams', 'Favorite'];
    final List<Widget> _children = [
    //playerPage2(context),
    Image.asset('assets/images/storvreta.jpg'),
    teamPage(context),
    Center(child: Text('No ads for 20kr'))
    
    ];
    return Scaffold(
      appBar: AppBar(
        //title: const Text('BottomNavigationBar Sample'),
        title: Text(_pageTitle[_selectedIndex]),
        actions: <Widget>[
          IconButton(
            icon: Icon(Icons.search),
            onPressed: () {
              showSearch(
                context: context,
                //delegate: CustomSearchDelegate(),
              );
            },
          ),
        ],
      ),
      //body: Center(
      // child: _children.elementAt(_selectedIndex),
      //),
      body: _children.elementAt(_selectedIndex),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            title: Text('Live'),
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.star),
            title: Text('Teams'),
          ),
          BottomNavigationBarItem(
            icon: ImageIcon(
              AssetImage('assets/images/storvreta3.jpg'),
              //color: Color(0xFF3A5A98),
            ),
            title: Text('Favorite'),
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.amber[800],
        onTap: _onItemTapped,
      ),
      drawer: drawer(context),
      // DrawerExample(onTap: (int val){
      //    setState(() {
      //    this._selectedIndexDrawer=val;
      // });
      // },)
    );
  }
}
