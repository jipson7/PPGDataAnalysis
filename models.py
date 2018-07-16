from flask_sqlalchemy import SQLAlchemy
import datetime
import pandas as pd

db = SQLAlchemy()


def extract_sensor_data(data, motion=False):
    index = []
    columns = ['red', 'ir', 'oxygen', 'hr', 'ratio', 'correlation']
    motion_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    rows = []
    for datum in data:
        index.append(datum.timestamp)
        reading = datum.reading
        algos = datum.algorithms
        row = [
            reading['red'],
            reading['ir']
        ]
        if algos is not None:
            enhanced = algos['enhanced']
            row += [
                enhanced['oxygen'] if enhanced['oxygen_valid'] == 1 else None,
                enhanced['hr'] if enhanced['hr_valid'] == 1 else None,
                enhanced['ratio'],
                enhanced['correlation']
            ]
        else:
            row += [None] * 4
        if motion:
            gyro = reading['gyro']
            accel = reading['accel']
            row += [
                gyro['x'],
                gyro['y'],
                gyro['z'],
                accel['x'],
                accel['y'],
                accel['z']
            ]
        rows.append(row)
    if motion:
        columns += motion_columns
    return pd.DataFrame(rows, index=index, columns=columns)


class Trial(db.Model):
    id = db.Column(db.INTEGER, primary_key=True)
    created = db.Column(db.TIMESTAMP, default=datetime.datetime.utcnow)
    user = db.Column(db.JSON)
    info = db.Column(db.TEXT)
    data = db.relationship("Data", order_by="Data.timestamp", lazy='dynamic')

    def __init__(self, user, info):
        self.user = user
        self.info = info

    @property
    def serialized(self):
        return {
            'id': self.id,
            'user': self.user,
            'info': self.info,
            'created': int(self.created.timestamp() * 1000)
        }

    def __str__(self):
        return str(self.id) + " " + \
               self.user['name'] + " - " + \
               self.info + " - " + \
               str(self.created)

    @property
    def df_wrist(self):
        return extract_sensor_data(self.wrist_data, motion=True)

    @property
    def df_reflective(self):
        return extract_sensor_data(self.reflective_data, motion=False)

    @property
    def df_transitive(self):
        return 0

    @property
    def wrist_data(self):
        return self.data.filter_by(device=0).all()

    @property
    def reflective_data(self):
        return self.data.filter_by(device=1).all()

    @property
    def transitive_data(self):
        return self.data.filter_by(device=2).all()


class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.TIMESTAMP)
    reading = db.Column(db.JSON)
    algorithms = db.Column(db.JSON)
    device = db.Column(db.Integer)
    trial_id = db.Column(db.Integer, db.ForeignKey('trial.id'))

    def __init__(self, timestamp, reading, device, trial_id):
        self.timestamp = datetime.datetime.fromtimestamp(timestamp / 1000.0)  # from ms
        self.reading = reading
        self.device = device
        self.trial_id = trial_id

    @property
    def serialized(self):
        return {
            'timestamp': int(self.timestamp.timestamp() * 1000),  # to ms
            'reading': self.reading,
            'algorithms': self.algorithms,
            'device': self.device
        }