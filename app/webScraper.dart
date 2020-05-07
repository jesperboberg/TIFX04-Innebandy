import 'package:flutter/material.dart' as mat;
import 'package:http/http.dart'; // Contains a client for making API calls
import 'package:html/parser.dart'; // Contains HTML parsers to generate a Document object
import 'package:html/dom.dart';

class WebScraper extends mat.StatefulWidget{
  @override
  _WebScraperState createState() => _WebScraperState();  
}

class _WebScraperState extends mat.State<WebScraper> {
  
  
  @override
  mat.Widget build(mat.BuildContext context) {
    initiate();
    
  List<String> toplist=SSLResponse.getInstance();
  List<mat.Widget> tiles = new List();
  int position = 0;
  for (String str in toplist)  {
    if(position < 14){
    position = position + 1;
    tiles.add(mat.ListTile(
      title: mat.Text('$position      ' + str),
    ));
    }
  }
  return mat.Scaffold(
    appBar: mat.AppBar(
                title: mat.Text('Tabell SSL dam'),
          
              ),
    body: mat.ListView(children: tiles),);
  }

}

Future <List<String>> initiate() async {
  List<String> teamnames = new List<String>();
  var client = Client();
  Response response = await client.get('https://ssl.se');

  // Use html parser
  var document = parse(response.body);

 List<Element> links = document.querySelectorAll(' td.rmss_t-team-statistics__name > a > span.rmss_t-team-statistics__name-short');
  List<Map<String, dynamic>> linkMap = [];

  for (var link in links) {
    linkMap.add({
      'teamName': link.text,
      
    });
  }
  for (var i = 0; i <linkMap.length; i++){
    teamnames.add(linkMap[i]['teamName']);
  }
  return teamnames;
}


class SSLResponse {
static List<String> _toplist;

static Future<bool> fetchTeams() async {
  _toplist = await initiate();
  return true;
}

static getInstance(){
  return _toplist;
}
}
