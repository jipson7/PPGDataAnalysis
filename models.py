from flask_sqlalchemy import SQLAlchemy
import datetime
import pandas as pd

db = SQLAlchemy()


class Trial(db.Model):
    id = db.Column(db.INTEGER, primary_key=True)
    created = db.Column(db.TIMESTAMP)
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
        data = self.wrist_data
        print(data[0].serialized)
        return "Loading"

    @property
    def df_reflective(self):
        return 0

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