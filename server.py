from flask import Flask, request
from models import db, Trial, Data
from sqlalchemy.exc import SQLAlchemyError
import logging
from flask import jsonify

app = Flask(__name__)
# TODO migrate config to file and DB to GCloud
app.config.from_object('config')
db.init_app(app)

# # Set log level to error
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)


@app.route('/')
def home():
    return app.send_static_file('index.html')


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


@app.route('/trials/<int:trial_id>/chart', methods=['GET'])
def get_chart_data(trial_id):
    device = int(request.args.get('device'))
    data_type = request.args.get('data')
    trial = Trial.query.get(trial_id)

    if device == 0:
        data = trial.wrist_data
    elif device == 1:
        data = trial.reflective_data
    elif device == 2:
        data = trial.transitive_data
    else:
        raise Exception("Invalid device type set in get parameter, select from 0, 1, or 2")

    result = []

    for datum in data:
        datum = datum.serialized
        reading = datum['reading']
        if data_type in reading:
            result.append({
                "x": datum.get("timestamp"),
                "y": reading.get(data_type)
            })
    return jsonify(data=result)


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
