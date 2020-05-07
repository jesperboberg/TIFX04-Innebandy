import 'package:flutter/material.dart';
import 'package:innebandy_test1/readFile.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:pie_chart/pie_chart.dart';

Widget playerPage2(BuildContext context, List<Player> players, String teamname) {
  return ListView.builder(
    itemCount: players.length+1,
    itemBuilder:(context, index){
      if (index==0){
       
      return Image.asset(
            'assets/images/$teamname.jpg',
            width: 600,
            height: 240,
            fit: BoxFit.cover,
          );}
      else{
      return playerTile(players.elementAt(index-1), context);}
    } 
    );
}

Widget playerTile(Player player, context) {
  return Card(
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30.0),),
    elevation: 5,
    child: ListTile(
      title: Text(player.name,
          style: TextStyle(
            fontWeight: FontWeight.w500,
            fontSize: 20,
          )),
      subtitle: Text(' number: ${player.nr}'),
      leading: Icon(IconData(59389, fontFamily: 'MaterialIcons')),
      trailing: FavoriteWidget(player),
      onTap: () {
        Navigator.push(context,
            new MaterialPageRoute(builder: (ctxt) => new Statspage(player)));
      },
    ),
  );
}

class Statspage extends StatelessWidget {
  final Player pl;
  Statspage(this.pl);

  @override
  Widget build(BuildContext context) {
    Map<String, double> dataMap = new Map();
    dataMap.putIfAbsent("Standing", () => pl.standPart.toDouble());
    dataMap.putIfAbsent("Walking", () => pl.walkPart.toDouble());
    dataMap.putIfAbsent("Jogging", () => pl.jogPart.toDouble());
    dataMap.putIfAbsent("Running", () => pl.runPart.toDouble());
    Widget titleSection = Container(
      padding: const EdgeInsets.all(20),
      child: Row(
        children: [
          Expanded(
            /*1*/
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                /*2*/
                Container(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(
                    '${pl.name}',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Text(
                  '${pl.team}',
                  style: TextStyle(
                    color: Colors.grey[500],
                  ),
                ),
              ],
            ),
          ),
          FavoriteWidget(pl),
        ],
      ),
    );


    Widget textSection = Container(
      padding: const EdgeInsets.all(20),
      child: Text(
        '${pl.name} har nummer ${pl.nr} och spelar i ${pl.team}.'
        ' Under denna match har ${pl.name} sprungit ${pl.distance} meter. ',
        softWrap: true,
      ),
    );

    return Scaffold(
      
      appBar: AppBar(
        title: Text('${pl.name}'),
      ),
      body: ListView(
        children: [
          Image.asset(
            'assets/images/heatmap.jpg',
            width: 600,
            height: 240,
            fit: BoxFit.cover,
          ),
          titleSection,
          textSection,
          PieChart(
      dataMap: dataMap,

      animationDuration: Duration(milliseconds: 800),
      chartLegendSpacing: 32.0,
      chartRadius: MediaQuery
          .of(context)
          .size
          .width / 2.7,
      showChartValuesInPercentage: true,
      showChartValues: true,
      ),
        ],
      ),
    );
  }

}

class FavoriteWidget extends StatefulWidget {
  final Player pl;
  FavoriteWidget(this.pl);
  @override
  _FavoriteWidgetState createState() => _FavoriteWidgetState(pl);
}

class _FavoriteWidgetState extends State<FavoriteWidget> {
  final Player player;
  _FavoriteWidgetState(this.player);
  @override
  Widget build(BuildContext context) {
    Player pl = player;
    bool _isFavorited;
    Future<dynamic> _future=SharedPreferenceHelper.getFavoritesSF();
    return Container(
      padding: EdgeInsets.all(0),
      child: FutureBuilder<dynamic>(
          future: _future,
          builder: (BuildContext context, AsyncSnapshot<dynamic> snapshot) {
            if (snapshot.data == true) {
              if (SharedPreferenceHelper.favoriteList.contains(pl.name))
                _isFavorited=true;
              else  
                _isFavorited=false;
              return IconButton(
                icon: (_isFavorited
                    ? Icon(Icons.favorite)
                    : Icon(Icons.favorite_border)),
                color: Colors.red[500],
                onPressed: () {
                  setState(() {
                    if (_isFavorited) {
                      
                      SharedPreferenceHelper.removeFromSF(pl.name);
                    } else{
                      SharedPreferenceHelper.addToFavoritesSF(pl.name);
                    }
                  });
                },
              );
            } else {
              return Icon(IconData(0xe7e9, fontFamily: 'MaterialIcons'));
            }
          }),
    );
  }
}

class SharedPreferenceHelper {
  static List<String> favoriteList = new List<String>();

  static Future<bool> addToFavoritesSF(String str) async {
    favoriteList.add(str);
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setStringList('favorite', favoriteList);
    return true;
  }

  static Future<bool> removeFromSF(String str) async {
    favoriteList.remove(str);
    print(favoriteList.length);
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setStringList('favorite', favoriteList);
    return true;
  }

  static Future<bool> getFavoritesSF() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    favoriteList = prefs.getStringList('favorite');
    return true;
  }

  static void resetSF() async {
    favoriteList.clear();
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setStringList('favorite', favoriteList);
  }
}
