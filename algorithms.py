from server import app, db, Data
from itertools import islice
from subprocess import run, PIPE
import json


def apply_algorithms_to_trial(trial_id):
    with app.app_context():
        base_query = Data.query.filter_by(trial_id=trial_id).order_by(Data.timestamp)
        wrist_data = base_query.filter_by(device=0).all()
        apply_algo_to_data_list(wrist_data)

        fingertip_data = base_query.filter_by(device=1).all()
        apply_algo_to_data_list(fingertip_data)


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