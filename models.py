from flask_sqlalchemy import SQLAlchemy
import datetime

db = SQLAlchemy()


class Trial(db.Model):
    id = db.Column(db.INTEGER, primary_key=True)
    created = db.Column(db.TIMESTAMP)
    user = db.Column(db.JSON)
    info = db.Column(db.TEXT)

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


class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.TIMESTAMP)
    reading = db.Column(db.JSON)
    device = db.Column(db.Integer)
    trial_id = db.Column(db.Integer, db.ForeignKey('trial.id'))
    trial = db.relationship(Trial, backref="data")

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
            'device': self.device
        }