import argparse
import json
import pickle as pkl

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


ICU_CODES = ['32057']
VENTILATION_CODES = [
    '4202832', '45887795', '765576', '37116689', '44791135', '4080957', '4072633',
    '2788038', '4230167', '4332501', '4235361', '37206832', '4251737', '2788037',
    '4168966', '40481547', '42738853', '44509482', '42738852', '40487536', '2788036',
    '4232550', '44515633', '2008008', '2314036', '2008007', '2314003', '2514578',
    '2106469', '2314000', '4337618', '4337047', '4202819', '4055375', '4072503',
    '4074667', '4337048', '4072505', '4080896', '4058031', '4174085', '4208272',
    '4055374', '4085542', '4335481', '4173351', '4072517', '4113618', '4236738',
    '4335585', '4072504', '4097246', '4283075', '4245036', '4072507', '4072515',
    '4072520', '4072522', '4331311', '4072516', '4308797', '4055377', '2788021',
    '2788019', '2788025', '4304419', '4055262', '3197551', '46273524', '4180290',
    '2007912', '2314001', '2314002', '2314035', '4013354', '4134853', '4244053',
    '4337045', '4072506', '40489935', '4283807', '4296607', '4134538', '4120570',
    '4303945', '4072521', '4337046', '4177224', '4057263', '4168475', '4074666',
    '4335584', '4026054', '4055261', '36676550', '4074669', '4287922', '4149878',
    '4140765', '4082243', '4055379', '4055376', '4219631', '44790095', '4337615',
    '4056812', '2788016', '2788024', '2788026', '2787824', '2745447', '2788020',
    '2787823', '4305389', '40488414', '4042360', '4165535', '4327886', '42535241',
    '2106470', '2106642', '2008006', '2008009', '4254209', '4179373', '4074670',
    '4072523', '4164571', '37116698', '4072519', '4072518', '4335583', '4287921',
    '4072631', '40486624', '4074668', '4055378', '4337616', '4074665', '4119642',
    '4337617', '4339623', '4237460', '4174555', '4229907', '2788022', '2788017',
    '2788027', '2745444', '2745440', '4039924', '3196459'
]


def create_notes_df(infile):
    notes = pd.read_json(infile, lines=True)
    rows = []
    for row in notes.itertuples():
        for note in row.visits:
            for sd in note['section_data']:
                for rfe in sd['risk_factor_entity']:
                    rows.append(
                        [str(row.patient_id), note['note_id'], note['timestamp'],
                         note['note_title'], sd['section_header'], rfe['chunk'],
                         rfe['entity'], rfe['assertion'], rfe['snomed_id']])
    notes = pd.DataFrame(rows, columns=['pid', 'note_id', 'timestamp', 'note_title',
                                        'section_header', 'chunk', 'entity',
                                        'assertion', 'snomed_id'])
    return notes


def parse_time(x):
    return pd.to_datetime(x[:19], format='%Y-%m-%dT%H:%M:%S')


def create_df(infile):
    global data, time_dict
    data = json.load(open(infile, 'r'))
    rows = []
    for pid, pdict in tqdm(data.items()):
        race, ethnicity, sex = pdict['race'], pdict['ethnicity'], pdict['sex']
        for visit in pdict['visits']:
            visit_id, age = visit['visit_occurrence_id'], visit['age_at_encounter']
            visit_starttime = parse_time(visit['visit_start_datetime'])
            visit_los = visit['outcome_length_of_stay'] * 24
            for time_, time_dict in visit.items():
                if type(time_dict) == dict:
                    for code_type in ['condition', 'drug', 'measurement', 'procedure']:
                        if code_type + 's' in time_dict:
                            time_dict[code_type] = time_dict[code_type + 's']
                            del time_dict[code_type + 's']
                        time_dict[code_type] = [str(c) for c in time_dict[code_type]]
                    for oc in ['outcome_icu', 'outcome_ventilation']:
                        if oc in time_dict:
                            time_dict[oc] = time_dict[oc]
                        else:
                            time_dict[oc] = 'N/A'
                    if not('measurement_values') in time_dict:
                        time_dict['measurement_values'] = {}
                    rows.append(
                        [f"{pid}_{visit_id}", sex, ethnicity, race, age,
                         visit_starttime, time_, time_dict['condition'],
                         time_dict['procedure'], time_dict['measurement'],
                         time_dict['drug'],
                         list(time_dict['measurement_values'].items()),
                         time_dict['outcome_icu'], time_dict['outcome_ventilation'],
                         visit_los])
    data = pd.DataFrame(rows, columns=['pid_vid', 'sex', 'ethnicity', 'race', 'age',
                                       'visit_starttime', 'timestamp', 'conditions',
                                       'procedures', 'measurements', 'drugs',
                                       'measurement_values', 'curr_outcome_icu',
                                       'curr_outcome_ventilation', 'visit_los_hours'])
    data['hour'] = (data['timestamp'].progress_apply(parse_time) - data['visit_starttime']) / np.timedelta64(1, 'h')
    data = data.loc[(data['hour'] >= 0) & (data['hour'] <= data['visit_los_hours'])]
    data.drop(columns=['timestamp', 'visit_starttime'], inplace=True)
    return data


def aggregate_data(data, aggregation_interval):
    data['timestep'] = (data['hour'] // aggregation_interval).astype(int)
    data['visit_los_num_timesteps'] = data['visit_los_hours'] / aggregation_interval
    agg_dict = {c : 'first' for c in ['race', 'ethnicity', 'sex', 'age', 'visit_los_num_timesteps']}
    agg_dict.update({c : 'sum' for c in ['conditions', 'procedures', 'drugs', 'measurements', 'measurement_values']})
    agg_dict.update({'curr_outcome_icu' : 'max', 'curr_outcome_ventilation' : 'max'})
    data = data.groupby(['pid_vid', 'timestep']).agg(agg_dict).reset_index()
    for c in ['conditions', 'procedures', 'drugs', 'measurements']:
        data[c] = data[c].apply(lambda x : list(set(x)))

    # Aggregate each measurement's value using mean.
    def f(x):
        d = {t[0] : [] for t in x}
        for t in x:
            d[t[0]].append(t[1])
        return {k : np.mean(v) for k, v in d.items()}
    data['measurement_values'] = data['measurement_values'].apply(f)
    return data


# In progress
def add_missing_timesteps(data):
    timesteps = data.groupby('pid_vid').agg({'timestep' : list}).reset_index()
    timesteps['missing'] = timesteps['timestep'].apply(lambda x: set(range(int(np.min(x)), int(np.max(x)))) - set(x))


def generate_los_data(data, prediction_window):
    data = data.loc[data['timestep'] + 1 <= data['visit_los_num_timesteps']]  # Exclude timesteps where discharged.
    data['outcome_los'] = (data['visit_los_num_timesteps'] <= data['timestep'] + prediction_window).astype(int)
    # discharge=1
    return data


def create_windows(data, num_timesteps, outcome_col):
    global timesteps
    # Output a dataframe with one row per sample.
    # Columns: pid_vid, input_timesteps
    timesteps = data.groupby('pid_vid').agg({'timestep' : list}).reset_index()
    timesteps['timestep'] = timesteps['timestep'].apply(lambda ts: sorted(ts))

    def get_windows(ts):
        windows = []
        start = 0
        for i in range(len(ts)):
            if ts[i] + 1 < num_timesteps:
                continue
            for t in ts:
                if t >= ts[i] - num_timesteps + 1:
                    start = t
                    break
            window = ts[start:i + 1]
            if window:
                windows.append(window)
        return windows
    timesteps['input_timesteps'] = timesteps['timestep'].apply(get_windows)
    rows = []
    for row in timesteps.itertuples():
        rows += [[row.pid_vid, w] for w in row.input_timesteps]
    timesteps = pd.DataFrame(rows, columns=['pid_vid', 'input_timesteps'])

    timesteps['timestep'] = timesteps['input_timesteps'].apply(max)
    timesteps = timesteps.merge(data[['pid_vid', 'timestep', outcome_col]],
                                on=['pid_vid', 'timestep'], how='left')
    timesteps.drop(columns=['timestep'], inplace=True)
    return timesteps


def print_stats(data):
    print('No. of patients:', data['pid_vid'].apply(lambda x: x.split('_')[0]).nunique())
    print('No. of visits:', data['pid_vid'].nunique())
    print('Are all timesteps in a visit contiguous?')
    timesteps = data.groupby('pid_vid').agg({'timestep' : list}).reset_index()
    timesteps['missing'] = timesteps['timestep'].apply(lambda x: set(range(int(np.min(x)), int(np.max(x)))) - set(x))
    if timesteps['missing'].apply(len).max() == 0:
        print('Yes')
    else:
        print('No')
    print('Do all visits start with timestep 0?')
    if timesteps['timestep'].apply(lambda x: np.min(x)).max() == 0:
        print('Yes')
    else:
        print('No')
    l = timesteps['timestep'].apply(len)
    print('No. of timesteps per visit (min, med, max):', l.min(), l.median(), l.max())
    for c in ['conditions', 'procedures', 'drugs', 'measurements', 'measurement_values']:
        l = data[c].apply(len)
        print('No. of', c, 'per timestep (min, med, max):', l.min(), l.median(), l.max())


def generate_task_data_from_codes(data, prediction_window, task):
    task_codes = eval(task.upper() + '_CODES')
    data['codes'] = data['conditions'] + data['procedures'] + data['drugs'] + data['measurements']
    curr_outcome_col = f"curr_{task}_outcome"

    def f(x):
        for c in task_codes:
            if c in x:
                return True
        return False
    data[curr_outcome_col] = data['codes'].progress_apply(f)
    data.drop(columns=['codes'], inplace=True)

    first_outcome_timestep = data.loc[data[curr_outcome_col] == 1].groupby('pid_vid').agg(
        {'timestep' : 'min'}).reset_index().rename(columns={'timestep' : 'first_outcome_timestep'})
    data = data.merge(first_outcome_timestep, on='pid_vid', how='left')
    data = data.loc[(data.first_outcome_timestep.isna()) | (data.timestep <= data.first_outcome_timestep)]

    # Add outcome column.
    data['outcome_' + task] = (data['first_outcome_timestep'] <= data['timestep'] + prediction_window).astype(int)
    # Remove last timestep with (task) event.
    data = data.loc[(data.first_outcome_timestep.isna()) | (data.timestep < data.first_outcome_timestep)]
    return data


def create_risk_factors(notes):
    global risk_factors
    notes = notes.loc[notes.assertion == 'present']
    vc = notes.entity.value_counts().reset_index()
    vc = vc.loc[vc.entity > 100]
    notes = notes.loc[notes.entity.isin(vc['index'])]
    notes = notes[['pid', 'entity']].drop_duplicates()
    risk_factors = pd.get_dummies(notes, columns=['entity'])
    entities = sorted([col for col in risk_factors.columns if col.startswith('entity')])
    print('List of risk factors', entities)
    risk_factors['risk_factors'] = risk_factors[entities].values.tolist()
    risk_factors['risk_factors'] = risk_factors['risk_factors'].apply(lambda x: np.array(x))
    rows = []
    for pid, group in risk_factors.groupby('pid'):
        rows.append([pid, list(group.risk_factors.sum())])
    risk_factors = pd.DataFrame(rows, columns=['pid', 'risk_factors'])
    return risk_factors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='path to top patient data folder contaiing DOB cohorts')
    parser.add_argument('--task', default='los', type=str, help='pretrain/los/icu/ventilation')
    parser.add_argument('--nlp_infile', type=str, help='path to nlp features file or NA')

    args = parser.parse_args()
    data = create_df(args.infile)

    risk_factors = None
    generate_risk_factors = (args.nlp_infile.upper() not in ['NA', 'N/A'])
    if generate_risk_factors:
        notes = create_notes_df(args.nlp_infile)
        risk_factors = create_risk_factors(notes)

    for aggregation_interval in [24]:
        aggregated_data = aggregate_data(data, aggregation_interval)
        print_stats(aggregated_data)
        if generate_risk_factors:
            aggregated_data['pid'] = aggregated_data['pid_vid'].apply(lambda x: x.split('_')[0])
            aggregated_data = aggregated_data.merge(risk_factors, on='pid', how='left')
            aggregated_data.drop(columns=['pid'], inplace=True)
            default_rf = [0 for i in range(len(risk_factors['risk_factors'].iloc[0]))]
            ii = aggregated_data.risk_factors.isna()
            aggregated_data.loc[ii, 'risk_factors'] = aggregated_data.loc[ii, 'risk_factors'].apply(lambda x: default_rf)

        if args.task == 'pretrain':
            pkl.dump(aggregated_data, open('../data/' + args.infile.split('/')[-1][:-5] + '_aggr' + str(aggregation_interval) + '.pkl', 'wb'))
        else:
            for lookahead in [2, 3, 7]:  # 2, 3, 7
                if (args.task == 'icu') or (args.task == 'ventilation'):
                    task_data = generate_task_data_from_codes(aggregated_data, lookahead, args.task)
                elif args.task == 'los':
                    task_data = generate_los_data(aggregated_data, lookahead)
                for num_timesteps in [1, 2, 4, 8]:  # 1, 2, 4, 8
                    task_windows = create_windows(task_data, num_timesteps, 'outcome_' + args.task)
                    print('# samples', len(task_windows))
                    print('lookahead:', lookahead, 'num_timesteps:', num_timesteps, '% pos outcome:', 100 * task_windows['outcome_' + args.task].sum() / len(task_windows))
                    pkl.dump([task_data, task_windows], open('../data/' + args.task + '_' + args.infile.split('/')[-1][:-5] + '_aggr' + str(aggregation_interval) + '_lookahead' + str(lookahead) + '_numts' + str(num_timesteps) + '.pkl', 'wb'))
