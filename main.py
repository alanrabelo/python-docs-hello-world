from flask import Flask
from wtforms import StringField, validators
from HMM import HMM
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, validators
from HMM import HMM
app = Flask(__name__)
app.config['SECRET_KEY'] = 'myKey'


class LoginForm(FlaskForm):
    sentence = StringField('sentence', validators=[validators.length(min=3, message=(u'Little short for a sentence?'))])

hmm = HMM()
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    form = LoginForm()
    if form.is_submitted():
        sentence = form.sentence.data
        result = hmm.classify(form.sentence.data, 1)
        if result is None:
            return render_template('index.html', form=form, result='Try to type something')
        phrase = sentence.replace('.', ' .').replace(',', ' ,').replace('?', ' ?').replace('!', ' !').split(' ')
        phrase = list(filter(None, phrase))


        resultDescription = ''
        for i, word in enumerate(phrase):
            if i < len(phrase):
                label = result[i]
                resultDescription += str(word) + "_" + str(label) + ' '

        return render_template('index.html', form=form, result=resultDescription)

    return render_template('index.html', form=form, title='Type a text')



if __name__ == '__main__':
  app.run()
##test