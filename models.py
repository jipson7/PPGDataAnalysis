from flask_sqlalchemy import SQLAlchemy
import datetime
import pandas as pd
import math

db = SQLAlchemy()


def get_speed(v):
    return math.sqrt((v['x']**2) + (v['y']**2) + (v['z']**2))


def extract_sensor_data(data, motion=False):
    index = []
    columns = ['red', 'ir', 'oxygen', 'hr', 'ratio', 'correlation']
    motion_columns = ['gyro', 'accel']
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
            raise RuntimeError("Remeber to apply algorithms binary to Trial before analyzing.")
        if motion:
            gyro_speed = get_speed(reading['gyro'])
            accel_speed = get_speed(reading['accel'])
            row += [
                gyro_speed,
                accel_speed
            ]
        rows.append(row)
    if motion:
        columns += motion_columns
    return pd.DataFrame(rows, index=index, columns=columns)


def remove_duplicate_timestamps(df):
    return df[~df.index.duplicated(keep='first')]


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
        return str(self.id) + " - " + \
               self.user['name'] + " - " + \
               self.info + " - " + \
               str(self.created)

    @property
    def df_wrist(self):
        df = extract_sensor_data(self.wrist_data, motion=True)
        return remove_duplicate_timestamps(df)

    @property
    def df_reflective(self):
        df = extract_sensor_data(self.reflective_data, motion=False)
        return remove_duplicate_timestamps(df)

    @property
    def df_transitive(self):
        data = self.transitive_data
        index = []
        columns = ['oxygen', 'hr']
        rows = []
        for datum in data:
            index.append(datum.timestamp)
            reading = datum.reading
            row = [
                reading['oxygen'] if reading['oxygen_valid'] else None,
                reading['hr'] if reading['hr_valid'] else None
            ]
            rows.append(row)
        df = pd.DataFrame(rows, index=index, columns=columns)
        return remove_duplicate_timestamps(df)

    @property
    def wrist_data(self):
        return self.data.filter_by(device=0).all()

    @property
    def reflective_data(self):
        return self.data.filter_by(device=1).all()

    @property
    def transitive_data(self):
        return self.data.filter_by(device=2).all()

    # Prints information but also has the side effect of loading the lazy devices
    def get_info(self):
        print("\ndf_wrist")
        print("columns: " + str(self.df_wrist.columns.tolist()))
        print("shape  : " + str(self.df_wrist.shape))
        print("\ndf_reflective")
        print("columns: " + str(self.df_reflective.columns.tolist()))
        print("shape  : " + str(self.df_reflective.shape))
        print("\ndf_transitive")
        print("columns: " + str(self.df_transitive.columns.tolist()))
        print("shape  : " + str(self.df_transitive.shape))


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