import 'package:flutter/material.dart';
import 'drawer.dart';
class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}
class _HomeState extends State<Home> {

  @override
  Widget build(BuildContext context) {
    Widget textSection = Container(
      padding: const EdgeInsets.all(25),
      child: Text(
        'Välkommen till "InnebandyStatistik". Sätt igång genom att klicka på fliken med förstoringsglaset'
        ' för att komma till söksidan. Där kan du söka på den spelare eller det lag du vill läsa statistik om. ',
        softWrap: true,
      ),
    );
    return Scaffold(
      appBar: AppBar(
        title: Text('Hem'),
      ),
      drawer: drawer(context),
      body: ListView(
        children: [
      Image.asset('assets/images/innebandy.jpg'),
          textSection,
        ],
      ),
    );
  }
}
