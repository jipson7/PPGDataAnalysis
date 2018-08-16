from server import app
from models import Trial


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)


def load_devices(trial_id):
    with app.app_context():
        trial = Trial.query.get(trial_id)
        trial.get_info()
        return [trial.df_wrist, trial.df_reflective, trial.df_transitive]


if __name__ == '__main__':
    list_trials()
    default_trial = 13
    devices = load_devices(default_trial)
