import 'package:flutter/material.dart';
import 'drawer.dart';

class AboutPage extends StatelessWidget {
  AboutPage(context);

  @override
  Widget build(BuildContext context) {
    final ThemeData themeData = Theme.of(context);
    final TextStyle aboutTextStyle = themeData.textTheme.body1;
    final TextStyle linkStyle =
        themeData.textTheme.body1.copyWith(color: themeData.accentColor);
    Widget titleSection = Container(
      padding: const EdgeInsets.all(32),
      child: Row(
        children: [
          Expanded(
            /*1*/
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                /*2*/
                Container(
                  child: Text(
                    "Kandidatarbete på Chalmers",
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),
          /*3*/
          Icon(
            Icons.star,
          ),
        ],
      ),
    );

    Widget textSection = Container(
      padding: const EdgeInsets.all(32),
      child: RichText(
        text: TextSpan(
          children: <TextSpan>[
            TextSpan(
              style: aboutTextStyle,
              text:
                  'Denna app är skapad av 6 studenter från Fysikteknologssektionen på Chalmers Tekniska Högskola. '
                  'Den presenterar statistik från innebandymatcher som tagits fram genom filmning och sedan spårning av '
                  'spelarnas rörelser.'
                  '\n \n'
                  'För att ta del av koden för projektet, klicka ',
            ),
            LinkTextSpan(
                style: linkStyle,
                url:
                    'https://github.com/jesperboberg/TIFX04-Innebandy/tree/master/app',
                text: 'här'),
            TextSpan(
              style: aboutTextStyle,
              text: '.',
            ),
          ],
        ),
      ),
    );

    return Scaffold(
      appBar: AppBar(
        title: Text('Kandidatarbete'),
      ),
      body: ListView(
        children: [
          Image.asset(
            'assets/images/chalmers.jpg',
            width: 600,
            height: 240,
            fit: BoxFit.cover,
          ),
          titleSection,
          textSection,
        ],
      ),
    );
  }
}
