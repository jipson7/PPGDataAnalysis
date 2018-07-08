from flask import Flask, request
from models import db, Trial, Data

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://caleb@localhost:5432/ppg'
db.init_app(app)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/trials', methods=['POST'])
def create_trial():
    json = request.get_json()
    user_json = json.get('user')
    trial_info = json.get('info')
    trial = Trial(user_json, trial_info)
    db.session.add(trial)

    try:
        db.session.commit()
        return str(trial.id)
    except Exception as e:
        print(e)
        return "Failed", 500


@app.route('/trials/<int:trial_id>', methods=['POST'])
def save_data(trial_id):
    json = request.get_json()
    reading = json.get('data')
    timestamp = json.get('timestamp')
    device = json.get('device')

    datum = Data(timestamp, reading, device, trial_id)

    db.session.add(datum)

    try:
        db.session.commit()
        return "Ok"
    except Exception as e:
        print(e)
        return "Failed", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
