# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sys import stdout
from sklearn.preprocessing import scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.utils.validation import indexable
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed


def compute_features():
    # Load json data
    with open('json_file.json') as data_file:
        patients = json.load(data_file)
    print("JSON file loaded")

    # Features computation
    print("Features computation launched")
    visits = []
    for patient in patients.values():
        for i in range(1, len(patient['visits']) + 1):
            visits.append(patient['visits'][str(i)])

    n_visits = len(visits)
    print("n_visits = %s" % n_visits)

    # Features DataFrame with encounter_nums index
    encounter_nums = [int(visit.get('encounter_num')) for visit in visits]
    X = pd.DataFrame(index=encounter_nums)

    # Time vector & censoring indicator
    print("Adding labels...", end="")
    next_visit = [visit.get('next_visit') for visit in visits]

    T = np.array([1e10 if str(t) == 'none' else t for t in next_visit]).astype(
        int)
    end_dates = pd.to_datetime([visit.get('end_date') for visit in visits])
    C = pd.to_datetime('2016-01-15 00:00:00') - end_dates
    days, seconds = C.days, C.seconds
    C = days * 24 + seconds // 3600  # in hours (discrete)
    delta = (T <= C).astype(int)
    Y = T
    Y[delta == 0] = C[delta == 0]
    labels = pd.DataFrame({'Y': Y, 'delta': delta}, index=encounter_nums)

    X = pd.concat([X, labels], axis=1)
    print(" done")

    # Basic features
    print("Adding basic features...", end="")
    # Add also patient_num & encounter_num for future random choice
    patient_num, encounter_num = [], []
    sex, baseline_HB, genotype_SS, age, transfu_count = [], [], [], [], []
    LS_ALONE, LS_INACTIVE, MH_ACS, MH_AVN, MH_DIALISIS = [], [], [], [], []
    MH_HEART_FAILURE, MH_ISCHEMIC_STROKE, MH_LEG_ULCER = [], [], []
    MH_NEPHROPATHY, MH_PHTN, MH_PRIAPISM, MH_RETINOPATHY = [], [], [], []
    OPIOID_TO_DISCHARGE, ORAL_OPIOID, USED_MORPHINE = [], [], []
    USED_OXYCODONE, duration, previous_visit, rea = [], [], [], []

    for patient in patients.values():
        for _ in range(1, len(patient['visits']) + 1):
            patient_num.append(patient['patient_num'])
            sex.append(1 if int(patient['sex']) == 1 else 0)
            baseline_HB.append(patient['baseline_HB'])
            genotype_SS.append(patient['genotype_SS'])

    for visit in visits:
        encounter_num.append(visit.get('encounter_num'))
        age.append(visit.get('age'))
        rea.append(visit.get('rea'))
        LS_ALONE.append(visit.get('LS_ALONE'))
        LS_INACTIVE.append(visit.get('LS_INACTIVE'))
        MH_ACS.append(visit.get('MH_ACS'))
        MH_AVN.append(visit.get('MH_AVN'))
        MH_DIALISIS.append(visit.get('MH_DIALISIS'))
        MH_HEART_FAILURE.append(visit.get('MH_HEART_FAILURE'))
        MH_ISCHEMIC_STROKE.append(visit.get('MH_ISCHEMIC_STROKE'))
        MH_LEG_ULCER.append(visit.get('MH_LEG_ULCER'))
        MH_NEPHROPATHY.append(visit.get('MH_NEPHROPATHY'))
        MH_PHTN.append(visit.get('MH_PHTN'))
        MH_PRIAPISM.append(visit.get('MH_PRIAPISM'))
        MH_RETINOPATHY.append(visit.get('MH_RETINOPATHY'))
        ORAL_OPIOID.append(visit.get('ORAL_OPIOID'))
        USED_MORPHINE.append(visit.get('USED_MORPHINE'))
        USED_OXYCODONE.append(visit.get('USED_OXYCODONE'))
        duration.append(visit.get('duration'))
        previous_visit.append(visit.get('previous_visit'))
        transfu_count.append(visit.get('transfu_count'))

    threshold = 24 * 30 * 18  # 18 months
    previous_visit = [0 if (t == 'none' or t > threshold) else 1 for t in
                      previous_visit]

    MH_ACS = [1 if int(x) == 2 else x for x in MH_ACS]
    MH_AVN = [1 if int(x) == 2 else x for x in MH_AVN]
    MH_DIALISIS = [1 if int(x) == 2 else x for x in MH_DIALISIS]
    MH_HEART_FAILURE = [1 if int(x) == 2 else x for x in MH_HEART_FAILURE]
    MH_ISCHEMIC_STROKE = [1 if int(x) == 2 else x for x in MH_ISCHEMIC_STROKE]
    MH_LEG_ULCER = [1 if int(x) == 2 else x for x in MH_LEG_ULCER]
    MH_NEPHROPATHY = [1 if int(x) == 2 else x for x in MH_NEPHROPATHY]
    MH_PHTN = [1 if int(x) == 2 else x for x in MH_PHTN]
    MH_PRIAPISM = [1 if int(x) == 2 else x for x in MH_PRIAPISM]
    MH_RETINOPATHY = [1 if int(x) == 2 else x for x in MH_RETINOPATHY]

    X_basic = pd.DataFrame(
        {'patient_num': patient_num, 'encounter_num': encounter_num, 'sex': sex,
         'genotype_SS': genotype_SS, 'age': age, 'rea': rea,
         'LS_INACTIVE': LS_INACTIVE, 'MH_ACS': MH_ACS, 'MH_AVN': MH_AVN,
         'MH_DIALISIS': MH_DIALISIS, 'MH_HEART_FAILURE': MH_HEART_FAILURE,
         'MH_ISCHEMIC_STROKE': MH_ISCHEMIC_STROKE,
         'MH_LEG_ULCER': MH_LEG_ULCER, 'LS_ALONE': LS_ALONE,
         'MH_NEPHROPATHY': MH_NEPHROPATHY, 'MH_PHTN': MH_PHTN,
         'MH_PRIAPISM': MH_PRIAPISM, 'MH_RETINOPATHY': MH_RETINOPATHY,
         'ORAL_OPIOID': ORAL_OPIOID, 'baseline_HB': baseline_HB,
         'USED_MORPHINE': USED_MORPHINE, 'USED_OXYCODONE': USED_OXYCODONE,
         'duration': duration, 'previous_visit': previous_visit,
         'transfu_count': transfu_count},
        index=encounter_nums)

    X = pd.concat([X, X_basic], axis=1)
    print(" done")

    # Bio data
    print("Adding bio features...", end="")
    bio_data, bio_names = pd.DataFrame(), []
    for visit in visits:
        encounter_num = int(visit.get('encounter_num'))
        tmp = pd.DataFrame(index=[encounter_num])
        end_date = pd.to_datetime(visit.get('end_date'))
        for bio_name, bio_values in visit.get('bio').items():

            # keep last value
            bio_names.append(bio_name)
            values = [val['nval_num'] for val in bio_values.values()]
            tmp[bio_name] = values[-1]

            # only keep last 48h values
            offset = end_date - pd.DateOffset(hours=48)
            values, index = [], []
            for dic in bio_values.values():
                val_time = pd.to_datetime(dic['date_bio'])
                if val_time > offset:
                    values.append(float(dic['nval_num']))
                    index.append(float(
                        (val_time - offset) / pd.Timedelta(
                            '1 hour')))

            # if at least 2 pts, add slope
            if len(values) > 1:
                x, y = index, values
                # least-squares
                A = np.vstack([np.array(x), np.ones(len(x))]).T
                slope, _ = np.linalg.lstsq(A, y)[0]
            else:
                slope = np.nan

            bio_names.append(bio_name + ' slope')
            tmp[bio_name + ' slope'] = slope

        bio_data = bio_data.append(tmp)

    bio_names_count = pd.Series(
        bio_names).value_counts() * 100 / n_visits
    bio_percentage = 35
    bio_param_kept = bio_names_count[bio_names_count > bio_percentage]
    bio_data = bio_data[bio_param_kept.index]
    print(" done")

    X = pd.concat([X, bio_data], axis=1)

    # Vital parameters data
    print("\nAdding vital parameters features...")

    param_no_gp = ['Poids [kg]', 'Taille [cm]',
                   'Débit O2 [L/min]']

    param_gp = ['Fréquence cardiaque [bpm]',
                'Fréquence respiratoire [mvt/min]', 'PA max [mmHg]',
                'PA min [mmHg]', 'Température [°C]',
                'Saturation en oxygène [%]']

    plot_curves_for_visits = np.random.randint(1, n_visits + 1, 3)
    print("\nPlot Gaussian Processes learned for a few random sampled visits")

    vital_parameter_data = pd.DataFrame()
    count = 1
    for nb_visit, visit in enumerate(visits):
        stdout.write(
            "\rVisit %s / %s" % (count, n_visits))
        stdout.flush()
        end_date = pd.to_datetime(visit.get('end_date'))
        encounter_num = int(visit.get('encounter_num'))
        tmp = pd.DataFrame(index=[encounter_num])
        for vital_name, vital_values in visit.get(
                'vital_parameters').items():

            if vital_name in param_gp:

                # only keep last 48h values
                offset = end_date - pd.DateOffset(hours=48)
                values, index = [], []
                for dic in vital_values.values():
                    val_time = pd.to_datetime(dic['start_date'])
                    if val_time > offset:
                        values.append(float(dic['nval_num']))
                        index.append(float(
                            (val_time - offset) / pd.Timedelta(
                                '1 hour')))

                if len(values) > 0:
                    x, y = index, values

                    # least-squares
                    A = np.vstack([np.array(x), np.ones(len(x))]).T
                    slope, intercept = np.linalg.lstsq(A, y)[0]
                    vals = np.array([slope, np.mean(values)])
                    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2, (
                        1, 5)) + WhiteKernel()
                    gp = GaussianProcessRegressor(kernel=kernel,
                                                  n_restarts_optimizer=10)
                    gp.fit(np.atleast_2d(x).T, np.array(y) - np.mean(y))
                    vals = np.append(vals, gp.kernel_.theta)

                    if count in plot_curves_for_visits:
                        nb_points = 100
                        x_new = np.linspace(0, 48, nb_points)
                        y_ = gp.predict(np.atleast_2d(x_new).T)
                        y_ += np.mean(y)

                        plt.figure()
                        plt.plot(x, y, 'r.', markersize=10,
                                 label=u'Observations')
                        plt.plot(x_new, y_, 'b-', label=u'Prediction')
                        plt.xlabel('last 48h')
                        plt.ylabel('values')
                        plt.title(vital_name + ", encounter_num = %s"
                                  % encounter_num)
                        plt.legend()
                        plt.show()
                else:
                    vals = np.array([np.nan] * 5)

                columns = ["%s slope" % vital_name,
                           "%s mean" % vital_name,
                           "%s cst_kernel" % vital_name,
                           "%s length_scale_RBF" % vital_name,
                           "%s noise_level" % vital_name]

                vals = pd.DataFrame(np.atleast_2d(vals), columns=columns,
                                    index=[encounter_num])
                tmp = pd.concat([tmp, vals], axis=1)

            if vital_name in param_no_gp:

                if vital_name in ['Poids [kg]', 'Taille [cm]']:
                    values = []
                    for dic in vital_values.values():
                        values.append(float(dic['nval_num']))

                    columns = ["%s mean" % vital_name]
                    vals = pd.DataFrame(np.atleast_2d(np.mean(values)),
                                        columns=columns,
                                        index=[encounter_num])

                if vital_name == 'Débit O2 [L/min]':
                    values, index = [], []
                    for dic in vital_values.values():
                        val_time = pd.to_datetime(dic['start_date'])
                        values.append(float(dic['nval_num']))
                        index.append(val_time)

                    if len(values) > 0:
                        idx_pos = [idx if is_positive else -1 for
                                   idx, is_positive
                                   in enumerate(np.array(values) > 0)]

                        delay = (end_date - index[
                            np.max(idx_pos)]) / pd.Timedelta('1 hour')
                    else:
                        delay = (end_date - start_date) / pd.Timedelta('1 hour')

                    columns = ["Débit O2 [L/min] delay"]
                    if delay < 0:
                        delay = 0
                    vals = pd.DataFrame(np.atleast_2d(delay),
                                        columns=columns,
                                        index=[encounter_num])
                tmp = pd.concat([tmp, vals], axis=1)

        vital_parameter_data = vital_parameter_data.append(tmp)
        count += 1

    # add BMI
    bmi = vital_parameter_data["Poids [kg] mean"] / vital_parameter_data[
                                                        "Taille [cm] mean"] ** 2
    bmi *= 1e4
    bmi = pd.DataFrame(bmi, columns=['BMI'])

    vital_parameter_data = pd.concat([vital_parameter_data, bmi], axis=1)

    X = pd.concat([X, vital_parameter_data], axis=1)

    # Syringes data
    print("\nAdding syringes features...")

    syringes_data = pd.DataFrame()
    count = 1
    for nb_visit, visit in enumerate(visits):
        stdout.write(
            "\rVisit %s / %s" % (count, n_visits))
        stdout.flush()
        start_date = pd.to_datetime(visit.get('start_date'))
        end_date = pd.to_datetime(visit.get('end_date'))
        encounter_num = int(visit.get('encounter_num'))
        tmp = pd.DataFrame(index=[encounter_num])

        bolus_dosage, index, refactory_period = [], [], []
        duration, max_dosage = [], []
        val_time = 0
        nb_syringes = 0
        for _, dic in visit.get('syringes').items():
            nb_syringes += 1
            val_time = pd.to_datetime(dic['opioid_start'])
            bolus_dosage.append(float(dic['bolus_dosage']))
            duration.append(float(dic['duration']))
            max_dosage.append(float(dic['max_dosage']))
            refactory_period.append(float(dic['refactory_period']))
            index.append(
                float((val_time - start_date) / pd.Timedelta('1 hour')))

        if val_time != 0:
            OPIOID_TO_DISCHARGE.append((end_date - val_time) / pd.Timedelta(
                '1 hour'))
        else:
            OPIOID_TO_DISCHARGE.append((end_date - start_date) / pd.Timedelta(
                '1 hour'))

        if len(index) > 0:
            first_syringe_time = np.min(index)
            if first_syringe_time < 0:
                index += abs(first_syringe_time)

            # areas
            area_bolus_dosage = np.sum(
                np.array(duration) * np.array(bolus_dosage))
            area_max_dosage = np.sum(np.array(duration) * np.array(max_dosage))
            area_refactory_period = np.sum(
                np.array(duration) * np.array(refactory_period))

            # least-squares
            A = np.vstack([np.array(index), np.ones(len(index))]).T
            try:
                slope_bolus_dosage, intercept_bolus_dosage = \
                    np.linalg.lstsq(A, bolus_dosage)[0]
            except:
                slope_bolus_dosage, intercept_bolus_dosage = np.nan, np.nan
            try:
                slope_max_dosage, intercept_max_dosage = \
                    np.linalg.lstsq(A, max_dosage)[0]
            except:
                slope_max_dosage, intercept_max_dosage = np.nan, np.nan
            try:
                slope_refactory_period, intercept_refactory_period = \
                    np.linalg.lstsq(A, refactory_period)[0]
            except:
                slope_refactory_period = np.nan
                intercept_refactory_period = np.nan

            # delay between every syringe
            if len(index) > 3:
                time_inter_syringe = []
                for i in range(1, len(index)):
                    time_inter_syringe.append(index[i] - index[i - 1])
                # least-squares
                A = np.vstack([range(len(time_inter_syringe)),
                               np.ones(len(time_inter_syringe))]).T
                try:
                    slope_delay_between_syringe, _ = \
                        np.linalg.lstsq(A, time_inter_syringe)[0]
                except:
                    slope_delay_between_syringe = np.nan

            else:
                slope_delay_between_syringe = np.nan

            syringe_frequency = nb_syringes / float(
                (end_date - start_date) / pd.Timedelta('1 hour'))

            vals = np.array([area_bolus_dosage, area_max_dosage,
                             area_refactory_period,
                             slope_bolus_dosage, intercept_bolus_dosage,
                             slope_max_dosage, intercept_max_dosage,
                             slope_refactory_period, intercept_refactory_period,
                             slope_delay_between_syringe,
                             syringe_frequency])

        else:
            vals = np.array([np.nan] * 11)

        columns = ["area bolus dosage", "area max dosage",
                   "area refactory period", "slope bolus dosage",
                   "intercept bolus dosage", "slope max dosage",
                   "intercept max dosage", "slope refactory period",
                   "intercept refactory period", "slope delay between syringe",
                   "syringe frequency"]

        vals = pd.DataFrame(np.atleast_2d(vals), columns=columns,
                            index=[encounter_num])
        tmp = pd.concat([tmp, vals], axis=1)

        syringes_data = syringes_data.append(tmp)
        count += 1

    X = pd.concat([X, syringes_data], axis=1)

    OPIOID_TO_DISCHARGE = pd.DataFrame(
        {'OPIOID_TO_DISCHARGE': OPIOID_TO_DISCHARGE},
        index=encounter_nums)

    X = pd.concat([X, OPIOID_TO_DISCHARGE], axis=1)

    # drop outliers
    X = X[X["encounter_num"] != 2008581722]
    X = X[X["encounter_num"] != 2009640859]
    X = X[X["encounter_num"] != 2011819004]
    X = X[X["encounter_num"] != 2008734750]

    # drop non-sens columns
    X = X.drop([' 04. (02PNP) Polynucleaires neutrophiles'], axis=1)
    X = X.drop([' 11. (02LYP) Lymphocytes'], axis=1)
    X = X.drop([' 13. (02MOP) Monocytes'], axis=1)
    X = X.drop([' 09. (02PBP) Polynucleaires basophiles'], axis=1)
    X = X.drop([' 06. (02PEP) Polynucleaires eosinophiles'], axis=1)

    # Rename Columns
    current_names = [
        'LS_ALONE', 'LS_INACTIVE', 'MH_ACS', 'MH_AVN', 'MH_DIALISIS',
        'MH_HEART_FAILURE', 'MH_ISCHEMIC_STROKE', 'MH_LEG_ULCER',
        'MH_NEPHROPATHY', 'MH_PHTN', 'MH_PRIAPISM', 'MH_RETINOPATHY',
        'OPIOID_TO_DISCHARGE', 'ORAL_OPIOID', 'USED_MORPHINE', 'USED_OXYCODONE',
        'age', 'baseline_HB', 'duration', 'genotype_SS',
        'previous_visit', 'rea', 'sex',
        ' 05. (02HT) Hématocrite',
        ' 03. (02GR) Hématies', ' 07. (02TGMH) Teneur Globulaire Moyenne',
        ' 04. (02HB) Hémoglobine', ' 02. (02GB) Leucocytes',
        ' 06. (02VGM) Volume Globulaire Moyen',
        ' 04. (02PNP) Polynucleaires neutrophiles',
        'Nb of polynucleaires eosinophiles', 'Nb of lymphocytes',
        'Nb of polynucleaires neutrophiles', 'Nb of monocytes',
        ' 11. (02LYP) Lymphocytes', ' 13. (02MOP) Monocytes',
        ' 09. (02PBP) Polynucleaires basophiles',
        'Nb of polynucleaires basophiles', ' 16. (01CL) Chlorures',
        ' 18. (01CO2) CO2 Total', ' 14. (01NA) Sodium', ' 19. (01PT) Protéines',
        ' 24. (01DFG1) DFG estimé (MDRD)', ' 22. (01CRE1) Créatinine',
        ' 15. (01K) Potassium', ' 3. (02VPM) Volume Plaquettaire Moyen',
        ' 11. (02RETAB) Réticulocytes', ' 49. (01BT) Bilirubine totale',
        ' 47. (01PAL) Phosphatases alcalines', ' 45. (01ASAT) ASAT',
        ' 10. (02PL) Plaquettes', ' 46. (01ALAT) ALAT',
        ' 06. (02PEP) Polynucleaires eosinophiles',
        ' 02. (01CRP) Protéine C-reactive', ' 48. (01GGT) Gamma GT',
        ' 50. (01BC) Bilirubine conjuguée', ' 20. (01CA) Calcium',
        ' 28. (01G) Glucose', ' 39. (02ERY) Erythroblastes',
        ' 41. (01LDH1) Lactate Deshydrogénase',
        ' 08. (02CCMH) Concentration corpusculaire moyenne en hémoglobine',
        ' 08. (01HBS) Hémoglobine S', ' 06. (01HBF) Hémoglobine F',
        ' 21. (01U) Urée',
        ' 05. (02HT) Hématocrite slope',
        ' 03. (02GR) Hématies slope',
        ' 07. (02TGMH) Teneur Globulaire Moyenne slope',
        ' 04. (02HB) Hémoglobine slope', ' 02. (02GB) Leucocytes slope',
        ' 06. (02VGM) Volume Globulaire Moyen slope',
        ' 04. (02PNP) Polynucleaires neutrophiles slope',
        'Nb of polynucleaires eosinophiles slope', 'Nb of lymphocytes slope',
        'Nb of polynucleaires neutrophiles slope', 'Nb of monocytes slope',
        ' 11. (02LYP) Lymphocytes slope', ' 13. (02MOP) Monocytes slope',
        ' 09. (02PBP) Polynucleaires basophiles slope',
        'Nb of polynucleaires basophiles slope', ' 16. (01CL) Chlorures slope',
        ' 18. (01CO2) CO2 Total slope', ' 14. (01NA) Sodium slope',
        ' 19. (01PT) Protéines slope',
        ' 24. (01DFG1) DFG estimé (MDRD) slope',
        ' 22. (01CRE1) Créatinine slope',
        ' 15. (01K) Potassium slope',
        ' 3. (02VPM) Volume Plaquettaire Moyen slope',
        ' 11. (02RETAB) Réticulocytes slope',
        ' 49. (01BT) Bilirubine totale slope',
        ' 47. (01PAL) Phosphatases alcalines slope', ' 45. (01ASAT) ASAT slope',
        ' 10. (02PL) Plaquettes slope', ' 46. (01ALAT) ALAT slope',
        ' 06. (02PEP) Polynucleaires eosinophiles slope',
        ' 02. (01CRP) Protéine C-reactive slope', ' 48. (01GGT) Gamma GT slope',
        ' 50. (01BC) Bilirubine conjuguée slope', ' 20. (01CA) Calcium slope',
        ' 28. (01G) Glucose slope', ' 39. (02ERY) Erythroblastes slope',
        ' 41. (01LDH1) Lactate Deshydrogénase slope',
        ' 08. (02CCMH) Concentration corpusculaire moyenne en hémoglobine slope',
        ' 08. (01HBS) Hémoglobine S slope', ' 06. (01HBF) Hémoglobine F slope',
        ' 21. (01U) Urée slope',
        'Débit O2 [L/min] delay',
        'Fréquence cardiaque [bpm] cst_kernel',
        'Fréquence cardiaque [bpm] mean',
        'Fréquence cardiaque [bpm] length_scale_RBF',
        'Fréquence cardiaque [bpm] noise_level',
        'Fréquence cardiaque [bpm] slope',
        'Saturation en oxygène [%] cst_kernel',
        'Saturation en oxygène [%] mean',
        'Saturation en oxygène [%] length_scale_RBF',
        'Saturation en oxygène [%] noise_level',
        'Saturation en oxygène [%] slope',
        'Fréquence respiratoire [mvt/min] cst_kernel',
        'Fréquence respiratoire [mvt/min] mean',
        'Fréquence respiratoire [mvt/min] length_scale_RBF',
        'Fréquence respiratoire [mvt/min] noise_level',
        'Fréquence respiratoire [mvt/min] slope', 'PA max [mmHg] cst_kernel',
        'PA max [mmHg] mean', 'PA max [mmHg] length_scale_RBF',
        'PA max [mmHg] noise_level', 'PA max [mmHg] slope',
        'PA min [mmHg] cst_kernel', 'PA min [mmHg] mean',
        'PA min [mmHg] length_scale_RBF', 'PA min [mmHg] noise_level',
        'PA min [mmHg] slope', 'Poids [kg] mean', 'Taille [cm] mean',
        'Température [°C] cst_kernel', 'Température [°C] mean',
        'Température [°C] length_scale_RBF', 'Température [°C] noise_level',
        'Température [°C] slope', 'BMI', 'area bolus dosage', 'area max dosage',
        'area refactory period', 'slope bolus dosage', 'mean bolus dosage',
        'slope max dosage', 'mean max dosage', 'slope refactory period',
        'mean refactory period', 'slope delay between syringe',
        'syringe frequency', 'transfu_count', ' 28. (01TCO2) CO2 Total',
        ' 30. (01HBGS) Hemoglobine', ' 27. (01SAO2) SaO2', ' 24. (01PH) pH',
        ' 32. (01HTEGS) Hematocrite', ' 26. (01PCO2) pCO2',
        ' 25. (01PO2) pO2', ' 14. (01TEMP) Temperature',
        ' 29. (01EB) Exces de bases', ' 3. (02TP%) TP', ' 4. (02INR) INR',
        " 4. (02INR) INR slope",
        " 3. (02TP%) TP slope",
        " 30. (01HBGS) Hemoglobine slope",
        " 27. (01SAO2) SaO2 slope",
        " 32. (01HTEGS) Hematocrite slope",
        " 25. (01PO2) pO2 slope",
        " 24. (01PH) pH slope",
        " 26. (01PCO2) pCO2 slope",
        " 28. (01TCO2) CO2 Total slope",
        " 14. (01TEMP) Temperature slope",
        " 29. (01EB) Exces de bases slope"]

    yo = pd.DataFrame()
    yo.corr()

    new_names = [
        'Household situation', 'Professional activity',
        'History of acute\nchest syndrom', 'History of avascular\nbone necrosis',
        'Formerly or currently\non a dialysis protocol',
        'History of heart\nfailure', 'History of\nischemic stroke',
        'History of leg\nskin ulceration', 'History of\nknown nephropathy',
        'History of pulmonary\nhypertension', 'History of priapism*',
        'History of known\nretinopathy',
        'Post-opioid observation\nperiod (hours)',
        'Received orally\nadministered opioids', 'Received Morphine',
        'Received Oxycodone', 'Age at hospital\nadmission',
        'Baseline haemoglobin\n($g/dL$)', 'Length of hospital\nstay (hours)',
        'Genotype', 'Less than 18 months\nsince last visit', 'Stayed in ICU',
        'Gender',
        'Hematocrit\n($\%$, mean)', 'Red blood cells\n($10^{12}/L$, mean)',
        'Mean corpuscular\nhemoglobin ($pg$, mean)',
        'Hemoglobin\n($g/dL$, mean)', 'White blood cells\n($10^9/L$, mean)',
        'Mean cell volume\n($fl$, mean)',
        'Neutrophils\n($\%$, mean)', 'Eosinophils\n($10^9/L$, mean)',
        'Lymphocytes\n($10^9/L$, mean)', 'Neutrophils\n($10^9/L$, mean)',
        'Monocytes\n($10^9/L$, mean)', 'Lymphocytes\n($\%$, mean)',
        'Monocytes\n($\%$, mean)', 'Basophils\n($\%$, mean)',
        'Basophils\n($10^9/L$, mean)', 'Chloride\n($mmol/L$, mean)',
        'Bicarbonate\n($mmol/L$, mean)',
        'Sodium\n($mmol/L$, mean)', 'Proteins\n($g/L$, mean)',
        'Renal function by MDRD\n($mL/min/1,73m2$, mean)',
        'Creatinine\n($\mu mol/L$, mean)', 'Potassium\n($mmol/L$, mean)',
        'Mean platelet volume\n($fl$, mean)', 'Reticulocytes\n($10^9/L$, mean)',
        'Total bilirubin\n($\mu mol/L$, mean)',
        'Alkaline phosphatase\n($U/L$, mean)',
        'Asparate transaminase\n($U/L, mean$)', 'Platelets\n($10^9/L$, mean)',
        'Alanine transaminase\n($U/L$, mean)', 'Eosinophils\n($\%$, mean)',
        'C-reactive protein\n($mg/L$, mean)',
        'Gamma glutamyl-tranferase\n($U/L$, mean)',
        'Direct bilirubin\n($\mu mol/L$, mean)',
        'Total calcium\n($mmol/L$, mean)', 'Glucose\n($mmol/L$, mean)',
        'Nucleated red blood\ncells ($10^9/L$, mean)',
        'Lactate Dehydrogenase\n($U/L$, mean)',
        'Mean corpuscular hemoglobin\nconcentration ($\%$, mean)',
        'Hemoglobin S\n($\%$, mean)', 'Hemoglobin F\n($\%$, mean)',
        'Urea ($mmol/L$, mean)',
        'Hematocrit (slope)', 'Red blood\ncells (slope)',
        'Mean corpuscular\nhemoglobin (slope)',
        'Hemoglobin (slope)', 'White blood\ncells (slope)',
        'Mean cell volume\n(slope)',
        'Neutrophils (slope)', 'Eosinophils (slope)',
        'Lymphocytes (slope)', 'Neutrophils (slope)',
        'Monocytes (slope)', 'Lymphocytes (slope)',
        'Monocytes (slope)', 'Basophils (slope)',
        'Basophils (slope)', 'Chloride (slope)',
        'Bicarbonate\n(slope)',
        'Sodium (slope)', 'Proteins (slope)',
        'Renal function by\nMDRD (slope)',
        'Creatinine (slope)', 'Potassium (slope)',
        'Mean platelet\nvolume (slope)', 'Reticulocytes\n(slope)',
        'Total bilirubin (slope)',
        'Alkaline phosphatase\n(slope)',
        'Asparate transaminase\n(slope)', 'Platelets (slope)',
        'Alanine transaminase\n(slope)', 'Eosinophils (slope)',
        'C-reactive protein\n(slope)',
        'Gamma glutamyl-\ntranferase (slope)',
        'Direct bilirubin\n(slope)',
        'Total calcium\n(slope)', 'Glucose (slope)',
        'Nucleated red blood\ncells (slope)',
        'Lactate Dehydrogenase\n(slope)',
        'Mean corpuscular hemoglobin\nconcentration (slope)',
        'Hemoglobin S (slope)', 'Hemoglobin F\n(slope)',
        'Urea (slope)',
        'Post-oxygen observation\nperiod (hours)',
        'Heart rate\n(constant kernel)', 'Heart rate\n(average)',
        'Heart rate (radial basis\nfunction kernel)',
        'Heart rate (noise\nlevel kernel)', 'Heart rate (slope)',
        'Oxygen saturation\n(constant kernel)', 'Oxygen saturation\n(average)',
        'Oxygen saturation (radial\nbasis function kernel)',
        'Oxygen saturation\n(noise level kernel)', 'Oxygen saturation\n(slope)',
        'Respiratory rate\n(constant kernel)', 'Respiratory rate\n(average)',
        'Respiratory rate (radial\nbasis ufnction kernel)',
        'Respiratory rate\n(noise)', 'Respiratory rate\n(slope)',
        'Systolic blood pressure\n(constant kernel)',
        'Systolic blood\npressure (average)',
        'Systolic blood pressure\n(radial basis function kernel)',
        'Systolic blood pressure\n(noise level kernel)',
        'Systolic blood\npressure (slope)',
        'Diastolic blood pressure\n(constant kernel)',
        'Diastolic blood\npressure (average)',
        'Diastolic blood pressure\n(radial basis function kernel)',
        'Diastolic blood pressure\n(noise level kernel)',
        'Diastolic blood\npressure (slope)', 'Weight ($kg$)', 'Size ($cm$)',
        'Temperature\n(constant kernel)', 'Temperature (average)',
        'Temperature (radial\nbasis function kernel)',
        'Temperature (noise\nlevel kernel)', 'Temperature (slope)',
        'Body mass index\n($kg/m^2$)', 'Bolus dosage (area)',
        'Maximum dosage (area)', 'Refractory period (area)',
        'Bolus dosage (slope)', 'Bolus dosage\n(average)',
        'Maximum dosage (slope)', 'Maximum dosage\n(average)',
        'Refractory period\n(slope)', 'Refractory period\n(average)',
        'Delay between syringes\n(slope)', 'Syringe frequency\n(per day)',
        'Transfusion count', 'ABG: total CO2\n(mmol/L)',
        'ABG: hemoglobin\n(g/dL)', 'ABG: oxygen\nsaturation (percent)', 'ABG: pH',
        'ABG: hematocrit\n(percent)', 'ABG: oxygen partial\npressure (mmHg)',
        'ABG: carbon dioxide\npartial pressure (mmHg)',
        'ABG: temperature\n(celcius)', 'ABG: base excess\n(mmol/L)',
        'Prothrombin Ratio\n(percent)', 'International\nNormalized Ratio',
        "International Normalized\nRatio (slope)",
        "Prothrombin Ratio\n(slope)",
        "ABG: hemoglobin (slope)",
        "ABG: oxygen saturation\n(slope)",
        "ABG: hematocrit (slope)",
        "ABG: carbon dioxide partial\npressure (slope)",
        "ABG: pH (slope)",
        "ABG: oxygen partial\npressure (slope)",
        "ABG: total CO2 (slope)",
        "ABG: Temperature\n(slope)",
        "ABG: base excess\n(slope)"]

    renamed_columns = list()
    print("\nUnchange column names:")
    for name in X.columns:
        idx = [i for i, x in enumerate(current_names) if x == name]
        try:
            renamed_columns.append(new_names[idx[0]])
        except:
            print(name)
            renamed_columns.append(name)

    X.columns = renamed_columns
    X.encounter_num = X.encounter_num.convert_objects(convert_numeric=True)
    X.patient_num = X.patient_num.convert_objects(convert_numeric=True)

    # Add GHM & ZIP infos
    GHM = pd.read_csv("../before_03_07_2017/GHM.csv", sep=";")
    GHM.CONCEPT_CD = GHM.CONCEPT_CD.apply(
        lambda x: x[-1].replace('T', '1').replace('Z', '1'))

    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_ghm = one_hot_encoder.fit_transform(np.atleast_2d(GHM.CONCEPT_CD).T)
    encoded_ghm = pd.DataFrame(encoded_ghm,
                               columns=["GHM=1", "GHM=2",
                                        "GHM=3", "GHM=4"])
    GHM = pd.concat([GHM, encoded_ghm], axis=1)
    GHM = GHM.drop('CONCEPT_CD', 1)
    X = X.merge(GHM, how="left", on='encounter_num')

    ZIP = pd.read_csv("../before_03_07_2017/ZIP.csv", sep=";")
    X = X.merge(ZIP, how="left", on='patient_num')

    X.to_csv("./all_visits.csv", sep=";")
    print("\nMatrix saved on disk")


def generate_data(verbose, encounter_nums=False):
    # Features generation
    if verbose:
        print("Features generation: pick 1 visit per patient randomly")

    # Load matrix with all visits
    X = pd.read_csv("./all_visits.csv", sep=";", index_col=0)

    if verbose:
        print("Matrix with all visits loaded")

    # Random choice of visit per patient
    X = X.groupby('patient_num').apply(
        lambda x: x.iloc[np.random.choice(range(0, len(x)))])

    X.index = X['encounter_num'].values
    X = X.drop('encounter_num', 1)
    X = X.drop('patient_num', 1)

    # Get labels
    Y = X['Y']
    X = X.drop('Y', 1)
    delta = X['delta']
    X = X.drop('delta', 1)

    n_visits = X.shape[0]
    if verbose:
        print("n_visits = %s" % n_visits)

    if verbose:
        print("Censoring rate: %.2f%%" % (100 * (1 - delta).sum() / n_visits))

    # replace strings 'nan' by numpy NaN
    X.replace('nan', np.nan, inplace=True)
    X.dist = [str(x).replace(',', '.') for x in X.dist.values]
    X.dist = X.dist.astype(float)
    X.drive = [str(x).replace(',', '.') for x in X.drive.values]
    X.drive = X.drive.astype(float)
    X = X.fillna(X.median())  # replace NaN by median

    # Time in days
    Y /= 24

    return X, np.array(Y), np.array(delta)

def cross_val_score_(estimators, X, y=None, groups=None, scoring=None,
                     cv=None, n_jobs=1, verbose=0, fit_params=None):
    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=True)
    cv_iter = list(cv.split(X, y, groups))

    parallel = Parallel(n_jobs=n_jobs, verbose=0)

    scores = parallel(delayed(_fit_and_score)(estimators[i], X, y,
                                              check_scoring(estimators[i],
                                                            scoring=scoring),
                                              train, test, verbose, None,
                                              fit_params)
                      for i, (train, test) in enumerate(cv_iter))

    return np.array(scores)[:, 0]


def compute_score(clf, X, y, K, verbose=True, fit_params=None):
    scores = cross_val_score_(clf, X, y, cv=K, verbose=0,
                              n_jobs=1, scoring="roc_auc",
                              fit_params=fit_params)
    score_mean = scores.mean()
    score_std = 2 * scores.std()
    if verbose:
        print("\n AUC: %0.3f (+/- %0.3f)" % (score_mean, score_std))
    return score_mean, score_std
