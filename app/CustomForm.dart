import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:innebandy_test1/writeFile.dart';

class MyCustomForm extends StatefulWidget {
  @override
  MyCustomFormState createState() {
    return MyCustomFormState();
  }
}

// Create a corresponding State class.
// This class holds data related to the form.
class MyCustomFormState extends State<MyCustomForm> {
  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<MyCustomFormState>.
  final _formKey = GlobalKey<FormState>();
  final teamController = TextEditingController();
  final nameController = TextEditingController();
  final numberController = TextEditingController();
  @override
  void dispose() {
    // Clean up the controller when the widget is disposed.
    teamController.dispose();
    nameController.dispose();
    numberController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.
    return Scaffold(
      body: Container(
          child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            teamFormField(teamController),
            nameFormField(nameController),
            numberFormField(numberController),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 16.0),
              child: RaisedButton(
                onPressed: () {
                  // Validate returns true if the form is valid, or false
                  // otherwise.
                  if (_formKey.currentState.validate()) {
                    CounterStorage();
                    print('hej jontefjonte');
                    // If the form is valid, display a Snackbar.
                    Scaffold.of(context).showSnackBar(
                        SnackBar(content: Text('Processing Data')));
                  }
                },
                child: Text('Submit'),
              ),
            ),
          ],
        ),
      )),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          return showDialog(
            context: context,
            builder: (context) {
              return AlertDialog(
                content: Text('Du har registrerat\n Lagnamn:' +
                    teamController.text +
                    '\n Spelarnamn' +
                    nameController.text +
                    '\n nummer:' +
                    numberController.text),
              );
            },
          );
        },
        tooltip: 'Show me the value!',
        child: Icon(Icons.text_fields),
      ),
    );
  }
}

TextFormField nameFormField(TextEditingController control) {
  return TextFormField(
    controller: control,
    decoration: const InputDecoration(
      icon: Icon(Icons.person),
      hintText: 'Vad heter spelaren?',
      labelText: 'Namn *',
    ),
  );
}

TextFormField numberFormField(TextEditingController control) {
  return TextFormField(
    controller: control,
    decoration: const InputDecoration(
      icon: Icon(Icons.star),
      hintText: 'Skriv numret på spelaren',
      labelText: 'Spelarens nummer *',
    ),
  );
}

TextFormField teamFormField(TextEditingController control) {
  return TextFormField(
    controller: control,
    decoration: const InputDecoration(
      icon: Icon(Icons.home),
      hintText: 'Vilket lag?',
      labelText: 'Lagnamn *',
    ),
  );
}

/*
TextFormField weightFormField() {
    return TextFormField(
      controller: _weightController,
      keyboardType: TextInputType.number,
      textInputAction: TextInputAction.done,
      focusNode: _weightFocus,
      onFieldSubmitted: (value){
        _weightFocus.unfocus();
        _calculator();
      },
      validator: (value) {
        if (value.length == 0 || double.parse(value) == 0.0) {
          return ('Weight is not valid. Weight > 0.0');
        }
      }, 
      onSaved: (value) {
        _weight = value;
      },
      decoration: InputDecoration(
          hintText: _weightMessage,
          labelText: _weightMessage,
          icon: Icon(Icons.menu),
          fillColor: Colors.white
      ),
    );
  }

  TextFormField heightFormField(BuildContext context) {
    return TextFormField(
      controller: _heightController,
      keyboardType: TextInputType.number,
      textInputAction: TextInputAction.next,
      focusNode: _heightFocus,
      onFieldSubmitted: (term) {
        _fieldFocusChange(context, _heightFocus, _weightFocus);
      },
      validator: (value) {
        if (value.length == 0 || double.parse(value) == 0.0) {
          return ('Height is not valid. Height > 0.0');
        }
      }, 
      onSaved: (value) {
        _height = value;
      },
      decoration: InputDecoration(
          hintText: _heightMessage,
          icon: Icon(Icons.assessment),
          fillColor: Colors.white,
      ),
    );
  }

  TextFormField ageFormField(BuildContext context) {
    return TextFormField(
      keyboardType: TextInputType.number,
      textInputAction: TextInputAction.next,
      focusNode: _ageFocus,
      onFieldSubmitted: (term){
        _fieldFocusChange(context, _ageFocus, _heightFocus);
      },
      validator: (value) {
        if (value.length == 0 || double.parse(value) <= 15) {
          return ('Age should be over 15 years old');
        }
      }, 
      onSaved: (value) {
        _age = value;
      },
      decoration: InputDecoration(
        hintText: 'Age',
        icon: Icon(Icons.person_outline),
        fillColor: Colors.white,
      ),
    );
  }*/
