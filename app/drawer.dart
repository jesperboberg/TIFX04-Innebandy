import 'package:flutter/material.dart';
//import 'package:flutter/gestures.dart';
//import 'package:url_launcher/url_launcher.dart';

Widget drawer(BuildContext context) {
  return Drawer(
    // Add a ListView to the drawer. This ensures the user can scroll
    // through the options in the drawer if there isn't enough vertical
    // space to fit everything.
    child: ListView(
      // Important: Remove any padding from the ListView.
      padding: EdgeInsets.zero,
      children: <Widget>[
        DrawerHeader(
          child: Text('Settings'),
          decoration: BoxDecoration(
            color: Colors.blue,
          ),
        ),
        ListTile(
          title: Text('Betygsätt appen'),
          onTap: () {
            Navigator.push(context,
                new MaterialPageRoute(builder: (ctxt) => new FirstPage()));
          },
        ),
        ListTile(
          title: Text('Kontakta oss'),
          onTap: () {
            Navigator.push(context,
                new MaterialPageRoute(builder: (ctxt) => new SecondPage()));
          },
        ),
        ListTile(
          title: Text('Om ApperApp'),
          onTap: () {
            Navigator.push(context,
                new MaterialPageRoute(builder: (ctxt) => new SecondPage()));
          },
        ),
      ],
    ),
  );
}

class FirstPage extends StatelessWidget {
  @override
  Widget build(BuildContext ctxt) {
    return new Scaffold(
      appBar: new AppBar(
        title: new Text("First Page"),
      ),
      body: new Text("I belongs to First Page"),
    );
  }
}

class SecondPage extends StatelessWidget {
  @override
  Widget build(BuildContext ctxt) {
    return new Scaffold(
      appBar: new AppBar(
        title: new Text("Second Page"),
      ),
      body: new Text("I belongs to Second Page"),
    );
  }
}

