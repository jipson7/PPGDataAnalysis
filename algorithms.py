from server import app, db, Trial
from itertools import islice
from subprocess import run, PIPE
import json


def apply_algorithms_to_trial(trial_id):
    with app.app_context():
        trial = Trial.query.get(trial_id)
        apply_algo_to_data_list(trial.wrist_data)
        apply_algo_to_data_list(trial.reflective_data)


def apply_algo_to_data_list(data_list):
    for window in windowized(data_list):
        last_data = window[-1]
        red_window = [str(x.reading.get('red')) for x in window]
        ir_window = [str(x.reading.get('ir')) for x in window]
        stdin = ','.join(red_window) + ' ' + ','.join(ir_window)
        p = run("bin/algos", stdout=PIPE, input=stdin, encoding='ascii')
        last_data.algorithms = json.loads(p.stdout)
    db.session.commit()


def windowized(seq, n=100):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

