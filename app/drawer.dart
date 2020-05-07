import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:innebandy_test1/aboutPage.dart';
import 'package:url_launcher/url_launcher.dart';
import 'betygPage.dart';
import 'themeChanger.dart';
import 'webScraper.dart';


Widget drawer(BuildContext context) {
  void _toDarkMode() {
    ThemeBuilder.of(context).changeTheme();
  }
  return Drawer(
    child: ListView(
      children: <Widget>[
        ListTile(
          title: Text('Om oss'),
          onTap: () {
            Navigator.push(context,
                MaterialPageRoute(builder: (context) => AboutPage(context)));
          },
        ),
        Divider(),
        ListTile(
          title: Text('Betygsätt appen'),
          onTap: () {
            Navigator.push(
                context, new MaterialPageRoute(builder: (ctxt) => RatePage()));
          },
        ),
        Divider(),
        
        
        ListTile(
          title: Text('SSL'),
          onTap: () {
            Navigator.push(context,
                new MaterialPageRoute(builder: (ctxt) => WebScraper()));
          },
        ),
        Divider(),
        ListTile(
          title: Text('Byt tema'),
          trailing: IconButton(
              icon: Icon(Icons.lightbulb_outline), onPressed: _toDarkMode),
        ),

        Divider(),
          ListTile(
            title: Text('Kontakta oss'),
            onTap: () {
              return showDialog(
                context: context,
                builder: (context) {
                  return AlertDialog(
                    title: new Text("Kontakta oss"),
                    content: new Text(
                        "Ni kan kontakta oss på denna mejl: innebandystatistik@gmail.com"),
                    actions: <Widget>[
                      new RaisedButton(
                        onPressed: () => _launchURL(
                            'innebandystatistik@gmail.com',
                            'InnebandyStatistik',
                            'Hej, '),
                        child: new Text('Skicka mail'),
                      ),
                      new FlatButton(
                        child: new Text("Stäng"),
                        onPressed: () {
                          Navigator.of(context).pop();
                        },
                      ),
                    ],
                  );
                },
              );
            },
          ),
      ],
    ),
  );
}

class LinkTextSpan extends TextSpan {
  LinkTextSpan({TextStyle style, String url, String text})
      : super(
            style: style,
            text: text ?? url,
            recognizer: TapGestureRecognizer()
              ..onTap = () {
                launch(url, forceSafariVC: false);
              });
}
_launchURL(String toMailId, String subject, String body) async {
  var url = 'mailto:$toMailId?subject=$subject&body=$body';
  if (await canLaunch(url)) {
    await launch(url);
  } else {
    throw 'Could not launch $url';
  }
} 
