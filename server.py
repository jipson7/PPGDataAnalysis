from flask import Flask, request, render_template
from models import db, Trial, Data
from sqlalchemy.exc import SQLAlchemyError
from flask import jsonify

app = Flask(__name__)
# TODO migrate config to file and DB to GCloud
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://caleb@localhost:5432/ppg'
db.init_app(app)


@app.route('/')
def home():
    return render_template('index.html')


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
    except SQLAlchemyError as e:
        print(e)
        return "Failed", 500


@app.route('/trials', methods=['GET'])
def get_trials():
    trials_result = Trial.query.all()
    trials = [t.serialized for t in trials_result]
    return jsonify(trials=trials)


@app.route('/trials/<int:trial_id>/data', methods=['GET'])
def get_trial_data(trial_id):
    trial = Trial.query.get(trial_id)
    data = [d.serialized for d in trial.data]
    return jsonify(data=data)


@app.route('/trials/<int:trial_id>/data', methods=['POST'])
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
    except SQLAlchemyError as e:
        print(e)
        return "Failed", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
