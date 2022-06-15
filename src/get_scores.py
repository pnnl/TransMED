import json
import sys


def get_scores(filepath, outcome_var):
    patient_score_dict = json.load(open(filepath))
    patient_scores_by_window = dict()
    #pid_max_ventilation = dict()
    for k, scores in patient_score_dict.items():
        ind = k.index(";")
        patient_id = k[0:ind]
        ts_data = k[ind+1:]
        patient_input_time_step = int(ts_data[0:ts_data.index(';')])
        #print(k, patient_id, patient_input_time_step)
        y_pred, y_true, y_score = scores[0], scores[1], scores[2]

        #if(patient_id not in pid_max_ventilation):
        #    pid_max_ventilation[patient_id] = y_true
        #else:
        #    pid_max_ventilation[patient_id] = max([y_true, pid_max_ventilation[patient_id]])
        if(patient_id not in patient_scores_by_window):
            patient_scores_by_window[patient_id] = list()
        #patient_scores_by_window[patient_id].append({'start_timestep': patient_input_time_step,\
        #                                     'score' :y_score, 'ventilation': y_true})
        patient_scores_by_window[patient_id].append({'start_timestep': patient_input_time_step,\
                                             'score' :y_score, outcome_var: y_true})
    return patient_scores_by_window


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Usage patient_score_dict, outcome_var")
        sys.exit(1)
    filepath = sys.argv[1]
    outcome_var = sys.argv[2]
    print(get_scores(filepath, outcome_var))
