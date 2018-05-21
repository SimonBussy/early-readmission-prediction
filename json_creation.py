# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json, re
from sys import stdout

# Visit data 
visit_drepano = pd.read_csv("../before_03_07_2017/data_init/CVO_stay.csv")
visit_drepano_init = visit_drepano.copy()

print("Visit data:")
print(visit_drepano.shape)
print(visit_drepano.head())

# Sort lines by date
visit_drepano.sort_values(["START_DATE", "END_DATE"], inplace=True)
visit_drepano.index = range(1, len(visit_drepano) + 1)

print(visit_drepano.head())

print("\n Finally : %s lines on the Visit dataset, corresponding to "
      "%s patients and %s different encounter_num" %
      (visit_drepano.shape[0], len(set(visit_drepano.PATIENT_NUM)),
       len(set(visit_drepano.ENCOUNTER_NUM))))

# Transfusion
transfusion = pd.read_csv("../before_03_07_2017/data_init/CVO_transfusion.csv",
                          sep=';',
                          encoding="ISO-8859-1")

print("\nTransfusion data:")
print(transfusion.shape)
print(transfusion.head())

visit_drepano = visit_drepano.merge(transfusion, how='outer', on="ENCOUNTER_NUM")
visit_drepano["transfu_count"].fillna(0, inplace=True)

# Bio data
# bio_drepano = pd.read_csv("../before_03_07_2017/data_init/CVO_bio.csv",
#                          sep=';', encoding="ISO-8859-1")

# including bios urgence
bio_drepano = pd.read_csv("../before_03_07_2017/data_init/CVO_bio_dont_sau.csv",
                          sep=';', encoding="utf-8")

print("\nBio data:")
print(bio_drepano.shape)
print(bio_drepano.head())

# # Get either tval_char or nval_num depending on the concept
# idx = bio_drepano.TVAL_CHAR != 'E'
# bio_drepano.NVAL_NUM[idx] = bio_drepano.TVAL_CHAR[idx]

# tval_char useless
bio_drepano.drop('TVAL_CHAR', axis=1, inplace=True)

# Remove lines with NA values
tmp = bio_drepano.shape[0]
bio_drepano = bio_drepano.replace('(null)', np.nan).dropna(axis=0, how='any')
print("%s lines removed" % (tmp - bio_drepano.shape[0]))

# Remove lines on the Visit dataset corresponding
# to patients without any bio info
tmp = visit_drepano.shape[0]
visit_drepano = visit_drepano[
    visit_drepano.PATIENT_NUM.isin(bio_drepano.PATIENT_NUM)]
print("%s lines removed in visit_drepano "
      "(without bio info)" % (tmp - visit_drepano.shape[0]))

# Remove lines on the Bio dataset not matching
# any patient_num of the Visit dataset
tmp = bio_drepano.shape[0]
tmp1 = (bio_drepano.PATIENT_NUM.unique()).shape[0]
bio_drepano = bio_drepano[
    bio_drepano.PATIENT_NUM.isin(visit_drepano.PATIENT_NUM)]
print("%s lines removed in bio_drepano (%s patients not in visit_drepano)" % (
    tmp - bio_drepano.shape[0],
    tmp1 - (bio_drepano.PATIENT_NUM.unique()).shape[0]))

# Remove lines on the Visit dataset not matching
# any encounter_num of the bio dataset
tmp = visit_drepano.shape[0]
visit_drepano = visit_drepano[
    visit_drepano.ENCOUNTER_NUM.isin(bio_drepano.ENCOUNTER_NUM)]
print("%s lines removed in visit_drepano"
      " (no encounter_num in bio_drepano)" % (tmp - visit_drepano.shape[0]))

# Remove lines on the bio dataset not matching
# any encounter_num of the Visit dataset
tmp = bio_drepano.shape[0]
bio_drepano = bio_drepano[
    bio_drepano.ENCOUNTER_NUM.isin(visit_drepano.ENCOUNTER_NUM)]
print("%s lines removed in bio_drepano"
      "(no encounter_num in visit_drepano)" % (tmp - bio_drepano.shape[0]))

# NAME_CHAR differ for same CONCEPT_CD depending on SOURCE (Adm ou Urg)
print("Merging different NAME_CHAR with same CONCEPT_CD...")
for concept in bio_drepano.CONCEPT_CD.unique():
    bio_drepano.NAME_CHAR.loc[bio_drepano.CONCEPT_CD == concept] = bio_drepano.NAME_CHAR.loc[bio_drepano.CONCEPT_CD == concept].iloc[0]

# Concepts to remove
concept_bio = bio_drepano[
    ['CONCEPT_CD', 'NAME_CHAR', 'PATIENT_NUM']].drop_duplicates().groupby(
    ['CONCEPT_CD', 'NAME_CHAR']).count()

# We only keep concepts present for at least 7 patients
concept2remove = [row[0] for row in map(list, concept_bio[
    concept_bio.PATIENT_NUM < 7].index.values)]

# Add the other concepts to remove
concept2remove += ['BIO:3771', 'BIO:4252', 'BIO:10392', 'BIO:10392', 'BIO:4249',
                   'BIO:16048', 'BIO:16044', 'BIO:16045', 'BIO:16049',
                   'BIO:16052', 'BIO:16053', 'BIO:16054', 'BIO:16055',
                   'BIO:4634', 'BIO:13576', 'BIO:1356', 'BIO:13577']

tmp = bio_drepano.shape[0]
tmp1 = bio_drepano.CONCEPT_CD.unique().shape[0]
bio_drepano = bio_drepano[~bio_drepano.CONCEPT_CD.isin(concept2remove)]
print("only keep concepts present for at least 7 patients:")
print("%s lines removed, corresponding to %s concepts" % (
    tmp - bio_drepano.shape[0],
    tmp1 - bio_drepano.CONCEPT_CD.unique().shape[0]))

# Name_char to merge
bio2replace = ['BIO:1008', 'BIO:3046', 'BIO:3762', 'BIO:3031', 'BIO:17069',
               'BIO:16987', 'BIO:3032', 'BIO:16612']
bio2keep = ['BIO:3066', 'BIO:17072', 'BIO:17068', 'BIO:17073', 'BIO:3056',
            'BIO:3056', 'BIO:17074', 'BIO:3029']

for i in range(len(bio2replace)):
    bio_drepano.NAME_CHAR = bio_drepano.NAME_CHAR.replace(
        bio_drepano[bio_drepano.CONCEPT_CD == bio2replace[i]].NAME_CHAR.values[0],
        bio_drepano[bio_drepano.CONCEPT_CD == bio2keep[i]].NAME_CHAR.values[0]
    )

# Name_char to clean manually
name2replace = ['.*PNAB.*', '.*PEAB.*', '.*PBAB.*', '.*LYAB.*', '.*MOAB.*']
value2replace = ['Nb of polynucleaires neutrophiles',
                 'Nb of polynucleaires eosinophiles',
                 'Nb of polynucleaires basophiles',
                 'Nb of lymphocytes', 'Nb of monocytes']

for i in range(len(name2replace)):
    bio_drepano.NAME_CHAR.replace(to_replace=name2replace[i],
                                  value=value2replace[i], inplace=True,
                                  regex=True)

print("Finally : %s bio infos remaining (%s distinct concepts), "
      "on %s different patients, with %s different encounter_num." %
      (len(bio_drepano.PATIENT_NUM), len(set(bio_drepano.NAME_CHAR)),
       len(set(bio_drepano.PATIENT_NUM)), len(set(bio_drepano.ENCOUNTER_NUM))))

# Demographic data
demography = pd.read_csv("../before_03_07_2017/data_init/CVO_patients.csv")

print("Patients data:")
print(demography.shape)
print(demography.head())

# Remove lines on the Demographic dataset not
# matching any patient_num of the Visit dataset
tmp = demography.shape[0]
demography = demography[demography.PATIENT_NUM.isin(visit_drepano.PATIENT_NUM)]
print("%s lines removed in demography"
      " (no patient_num in visit_drepano)" % (tmp - demography.shape[0]))

print("Finally : %s lines on the Demography dataset, "
      "corresponding to %s patients" % (
          len(demography.PATIENT_NUM), len(set(demography.PATIENT_NUM))))

# Vital_parameters data
vital_parameters = pd.read_csv("../before_03_07_2017/data_init/CVO_param.csv",
                               sep=';',
                               encoding="ISO-8859-1")
print("Vital parameters data:")
print(vital_parameters.shape)
print(vital_parameters.head())

# Remove lines on the Vital_parameters dataset
# not matching any patient_num of the Visit dataset
tmp = vital_parameters.shape[0]
vital_parameters = vital_parameters[
    vital_parameters.PATIENT_NUM.isin(visit_drepano.PATIENT_NUM)]
print("%s lines removed in vital_parameters"
      " (no patient_num in visit_drepano)" % (tmp - vital_parameters.shape[0]))

# Remove lines on the Vital_parameters dataset not
# matching any encounter_num of the Visit dataset
tmp = vital_parameters.shape[0]
vital_parameters = vital_parameters[
    vital_parameters.ENCOUNTER_NUM.isin(visit_drepano.ENCOUNTER_NUM)]
print("%s lines removed in vital_parameters"
      " (no encounter_num in visit_drepano)" % (
          tmp - vital_parameters.shape[0]))

# Gathering concepts
name2replace = ['Temp.*', 'Saturation.*', '.*Poids.*', '.*PA.*', '.*Oxyg.*',
                '.*respiratoire.*', '.*cardiaque.*', '.*EVA.*']
value2replace = ['Température [°C]', 'Saturation en oxygène [%]', 'Poids [kg]',
                 'PA syst/diast. [mmHg]', 'Débit O2 [L/min]',
                 'Fréquence respiratoire [mvt/min]',
                 'Fréquence cardiaque [bpm]', 'Douleur EVA']

for i in range(len(name2replace)):
    vital_parameters.NAME_CHAR.replace(to_replace=name2replace[i],
                                       value=value2replace[i], inplace=True,
                                       regex=True)

# Separate pression systolic & diastolic
tmp = vital_parameters[vital_parameters.NAME_CHAR == 'PA syst/diast. [mmHg]']
vital_parameters = vital_parameters[
    vital_parameters.NAME_CHAR != 'PA syst/diast. [mmHg]']
tmp = tmp[tmp.TVAL_CHAR.str.contains(' :: ')]
PAmax, PAmin = tmp.TVAL_CHAR.apply(
    lambda x: x.split(' :: ')[0]), tmp.TVAL_CHAR.apply(
    lambda x: x.split(' :: ')[1])
tmp.NAME_CHAR, tmp.TVAL_CHAR = 'PA max [mmHg]', PAmax
vital_parameters = vital_parameters.append(tmp, ignore_index=True)
tmp.NAME_CHAR, tmp.TVAL_CHAR = 'PA min [mmHg]', PAmin

vital_parameters.TVAL_CHAR = vital_parameters.TVAL_CHAR.apply(
    lambda x: x.replace(',', '.'))
vital_parameters = vital_parameters.rename(columns={'TVAL_CHAR': 'NVAL_NUM'})

print('Variables in the Vital_parameters dataset:')
for var in set(vital_parameters.NAME_CHAR):
    print(var)

print("Finally : %s visits have corresponding vital parameters data "
      "in the Vital_parameters dataset, among the %s visits remaining "
      "in the Visit dataset, corresponding to %s patients among the %s "
      "total present in the Visit dataset." %
      (len(set(vital_parameters.ENCOUNTER_NUM)),
       len(set(visit_drepano.ENCOUNTER_NUM)),
       len(set(vital_parameters.PATIENT_NUM)),
       len(set(visit_drepano.PATIENT_NUM))))

# Pancarte data
pancarte = pd.read_csv("../before_03_07_2017/data_init/CVO_pancarte.csv",
                       sep=';',
                       encoding="ISO-8859-1")
print("Pancarte data:")
print(pancarte.shape)
print(pancarte.head())

# Remove pain localisation because unstructured data
pancarte = pancarte[pancarte.CONCEPT_CD != 'QST:QN|16398']
pancarte = pancarte[pancarte.TVAL_CHAR != 'Evaluation impossible']
pancarte.TVAL_CHAR = pancarte.TVAL_CHAR.replace('Oui', 1).replace('Non', 0)

# Get either tval_char or nval_num depending on the concept
val_pancarte = [re.sub(r'.CM', '', str(row)) for row in pancarte.TVAL_CHAR]
for i, val in enumerate(pancarte.NVAL_NUM):
    if val != '(null)':
        val_pancarte[i] = str(val).replace(',', '.')
val_pancarte = pd.Series(val_pancarte, dtype=np.float64)

pancarte = pancarte.drop('TVAL_CHAR', 1)
pancarte.index = range(0, len(pancarte))
pancarte.NVAL_NUM = val_pancarte
pancarte = pancarte.dropna(axis=0)
pancarte[['PATIENT_NUM', 'ENCOUNTER_NUM']] = pancarte[
    ['PATIENT_NUM', 'ENCOUNTER_NUM']].astype(int)

# Remove lines on the Pancarte dataset not
# matching any patient_num of the Visit dataset
tmp = pancarte.shape[0]
pancarte = pancarte[pancarte.PATIENT_NUM.isin(visit_drepano.PATIENT_NUM)]
print("%s lines removed in pancarte"
      "(no patient_num in visit_drepano)" % (tmp - pancarte.shape[0]))

# Remove lines on the Pancarte dataset not
# matching any encounter_num of the Visit dataset
tmp = pancarte.shape[0]
pancarte = pancarte[pancarte.ENCOUNTER_NUM.isin(visit_drepano.ENCOUNTER_NUM)]
print("%s lines removed in pancarte"
      "(no encounter_num in visit_drepano)" % (tmp - pancarte.shape[0]))

# Put the same concept names that in the Vital_parameter dataset
name2replace = ['.*Temp.*', '.*SaO2.*', '.*Poids.*', '.*Max.*', '.*Min.*',
                '.*bit O2.*', '.*FR.*', '.*FC.*', '.*EVA.*', '.*Taille.*',
                '.*Patient.*']
value2replace = ['Température [°C]', 'Saturation en oxygène [%]', 'Poids [kg]',
                 'PA max [mmHg]', 'PA min [mmHg]', 'Débit O2 [L/min]',
                 'Fréquence respiratoire [mvt/min]',
                 'Fréquence cardiaque [bpm]', 'Douleur EVA', 'Taille [cm]',
                 'Sous O2 [0/1]']

for i in range(len(name2replace)):
    pancarte.NAME_CHAR.replace(to_replace=name2replace[i],
                               value=value2replace[i], inplace=True, regex=True)

print('Variables in the pancarte:')
for var in set(pancarte.NAME_CHAR):
    print(var)

print("Finally : %s visits have corresponding pancarte data, "
      "among the %s visits remaining in the Visit dataset, "
      "corresponding to %s patients among the %s total "
      "present in the Visit dataset." %
      (len(set(pancarte.ENCOUNTER_NUM)), len(set(visit_drepano.ENCOUNTER_NUM)),
       len(set(pancarte.PATIENT_NUM)), len(set(visit_drepano.PATIENT_NUM))))
print("\nAnd %s visits have corresponding at least one pancarte "
      "data or one vital parameter data, among the %s visits "
      "remaining in the Visit dataset, corresponding to %s patients "
      "among the %s total present in the Visit dataset." %
      (len(set(
          pd.concat([pancarte.ENCOUNTER_NUM, vital_parameters.ENCOUNTER_NUM],
                    axis=0).values)), len(set(visit_drepano.ENCOUNTER_NUM)),
       len(set(pd.concat([pancarte.PATIENT_NUM, vital_parameters.PATIENT_NUM],
                         axis=0).values)), len(set(visit_drepano.PATIENT_NUM))))

print(set(visit_drepano.ENCOUNTER_NUM) - set(
    pd.concat([pancarte.ENCOUNTER_NUM, vital_parameters.ENCOUNTER_NUM],
              axis=0).values))

print("\nSo %s patients don't have any data in the "
      "pancarte or vital parameter dataset:" %
      (len(set(visit_drepano.PATIENT_NUM)) - len(set(
          pd.concat([pancarte.PATIENT_NUM,
                     vital_parameters.PATIENT_NUM], axis=0).values))))

print(set(visit_drepano.PATIENT_NUM) - set(
    pd.concat([pancarte.PATIENT_NUM,
               vital_parameters.PATIENT_NUM], axis=0).values))

# Merging Pancarte & Vital_parameter data into a single Vital-parameter dataset
Vital_parameters = vital_parameters.append(pancarte, ignore_index=True)

# Syringes data
Syringes = pd.read_csv("../before_03_07_2017/data_init/CVO_seringues.csv",
                       sep=',',
                       encoding="ISO-8859-1")
print("\nSyringes data:")
print(Syringes.shape)
print(Syringes.head())

# Remove lines on the Syringes dataset
# not matching any patient_num of the Visit dataset
tmp = Syringes.shape[0]
Syringes = Syringes[
    Syringes.PATIENT_NUM.isin(visit_drepano.PATIENT_NUM)]
print("\n%s lines removed in syringes"
      " (no patient_num in visit_drepano)" % (tmp - Syringes.shape[0]))

# Remove lines on the Syringes dataset not
# matching any encounter_num of the Visit dataset
tmp = Syringes.shape[0]
Syringes = Syringes[
    Syringes.ENCOUNTER_NUM.isin(visit_drepano.ENCOUNTER_NUM)]
print("%s lines removed in syringes"
      " (no encounter_num in visit_drepano)" % (
          tmp - Syringes.shape[0]))

# Get all visits for previous_visit field
print("All visits:")

all_visits = pd.io.parsers.read_table("../before_03_07_2017/data_init/CVO_previous.csv",
                                      sep=',')
print("%s lines on the Visit dataset have "
      "previous visits (before 2010)." % len(all_visits))

# Patient object definition and JSON file creation
print("JSON file creation")


class CreateDict(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)


Patients = {}
list_patient_num = list(visit_drepano.PATIENT_NUM)

# list_patient_num = list_patient_num[:3]

consult = pd.read_csv("../before_03_07_2017/consult.csv", sep=";")


id_patient = 1
count = 1
for patient_num in sorted(set(list_patient_num),
                          key=lambda x: list_patient_num.index(x)):
    demog = demography.loc[demography.PATIENT_NUM == patient_num]
    birth_date = demog.BIRTH_DATE.values[0]
    sex = demog.SEX.values[0]
    death = demog.DEATH.values[0]
    death_date = demog.DEATH_DATE.values[0]
    ddn = demog.DDN_DATE.values[0]
    baseline_HB = demog.BASELINE_HB.values[0]
    genotype_SS = demog.GENOTYPE_SS.values[0]

    visits_associated = visit_drepano.loc[
        visit_drepano.PATIENT_NUM == patient_num]
    # use all visits to compute next and previous visit
    visits_associated_all = visit_drepano_init.loc[
        visit_drepano_init.PATIENT_NUM == patient_num]
    list_encounter_num = list(visits_associated.ENCOUNTER_NUM)

    visits = {}
    id_visit = 1
    for encounter_num in sorted(set(list_encounter_num),
                                key=lambda x: list_encounter_num.index(x)):

        stdout.write("\rVisit %s / %s" % (count, visit_drepano.shape[0]))
        stdout.flush()
        count += 1

        visit_detail = visits_associated[
            visits_associated.ENCOUNTER_NUM == encounter_num]
        rea = visit_detail.ICU_STAY.values[0]
        ORAL_OPIOID = visit_detail.ORAL_OPIOID.values[0]
        USED_MORPHINE = visit_detail.USED_MORPHINE.values[0]
        USED_OXYCODONE = visit_detail.USED_OXYCODONE.values[0]
        OPIOID_TO_DISCHARGE = visit_detail.OPIOID_TO_DISCHARGE.values[0]
        LS_INACTIVE = visit_detail.LS_INACTIVE.values[0]
        LS_ALONE = visit_detail.LS_ALONE.values[0]
        MH_ACS = visit_detail.MH_ACS.values[0]
        MH_PRIAPISM = visit_detail.MH_PRIAPISM.values[0]
        MH_AVN = visit_detail.MH_AVN.values[0]
        MH_ISCHEMIC_STROKE = visit_detail.MH_ISCHEMIC_STROKE.values[0]
        MH_LEG_ULCER = visit_detail.MH_LEG_ULCER.values[0]
        MH_HEART_FAILURE = visit_detail.MH_HEART_FAILURE.values[0]
        MH_PHTN = visit_detail.MH_PHTN.values[0]
        MH_RETINOPATHY = visit_detail.MH_RETINOPATHY.values[0]
        MH_NEPHROPATHY = visit_detail.MH_NEPHROPATHY.values[0]
        MH_DIALISIS = visit_detail.MH_DIALISIS.values[0]
        transfu_count = visit_detail.transfu_count.values[0]
        start_date = pd.to_datetime(visit_detail.START_DATE.values[0])
        end_date = pd.to_datetime(visit_detail.END_DATE.values[-1])
        age = (start_date - pd.to_datetime(birth_date)).days / 365.
        duration = end_date - start_date
        days, seconds = duration.days, duration.seconds
        duration = days * 24 + seconds / 3600.  # in hours

        try:
            next_visit = pd.to_datetime(
                visits_associated_all[pd.to_datetime(
                    visits_associated_all.START_DATE)
                                      >= end_date].START_DATE.values.min()) \
                         - end_date
            days, seconds = next_visit.days, next_visit.seconds
            next_visit = days * 24 + seconds // 3600  # in hours (discrete)
        except:
            next_visit = 'none'

        next_consult = consult.consult[consult.ENCOUNTER_NUM == encounter_num]
        next_consult = pd.to_datetime(next_consult.values.min())
        if not isinstance(next_consult, pd.tslib.NaTType):
            next_consult = next_consult - end_date
            days, seconds = next_consult.days, next_consult.seconds
            next_consult = days * 24 + seconds // 3600  # in hours (discrete)

            if next_visit != 'none' and next_visit > next_consult > 0:

                next_visit = next_consult

        try:
            yo = all_visits[all_visits.PATIENT_NUM == patient_num]
            previous_visit = start_date - pd.to_datetime(yo[pd.to_datetime(
                yo.END_DATE) <= start_date].END_DATE.values.max())
            days, seconds = previous_visit.days, previous_visit.seconds
            previous_visit = days * 24 + seconds / 3600.  # in hours
        except:
            previous_visit = 'none'

        bio = bio_drepano.loc[bio_drepano.ENCOUNTER_NUM == encounter_num]
        bio_data = {}
        id_bio = 1
        for name_char in set(bio.NAME_CHAR):
            bio_detail = bio[bio.NAME_CHAR == name_char].sort_values(
                by="DATE_BIO")
            bio_detail.index = range(1, len(bio_detail) + 1)
            val = {}
            for id_val in range(len(bio_detail)):
                nval_num = bio_detail.NVAL_NUM.values[id_val]
                # nval_num = float(nval_num.replace(',', '.'))
                val[str(id_val + 1)] = CreateDict(
                    nval_num=nval_num,
                    concept_cd=bio_detail.CONCEPT_CD.values[id_val],
                    date_bio=bio_detail.DATE_BIO.values[id_val]
                )
            bio_data[name_char] = val
            id_bio += 1

        vital_data = Vital_parameters.loc[
            Vital_parameters.ENCOUNTER_NUM == encounter_num]
        vital_param = {}
        id_vital = 1
        for name_char in set(vital_data.NAME_CHAR):
            vital_detail = vital_data[
                vital_data.NAME_CHAR == name_char].sort_values(by="START_DATE")
            vital_detail.index = range(1, len(vital_detail) + 1)
            val = {}
            for id_val in range(len(vital_detail)):
                val[str(id_val + 1)] = CreateDict(
                    nval_num=str(vital_detail.NVAL_NUM.values[id_val]),
                    concept_cd=vital_detail.CONCEPT_CD.values[id_val],
                    start_date=vital_detail.START_DATE.values[id_val]
                )
            vital_param[name_char] = val
            id_vital += 1

        syringes_data = Syringes.loc[
            Syringes.ENCOUNTER_NUM == encounter_num]
        syringes_data = syringes_data.sort_values(by="OPIOID_START")
        syringes_data.index = range(1, len(syringes_data) + 1)
        syringes = {}
        for id_val in range(len(syringes_data)):
            syringes[str(id_val + 1)] = CreateDict(
                opioid_start=str(syringes_data.OPIOID_START.values[id_val]),
                duration=syringes_data.DURATION.values[id_val],
                molecule=syringes_data.OPIOID_MOLECULE.values[id_val],
                bolus_dosage=syringes_data.BOLUS_DOSAGE.values[id_val],
                refactory_period=syringes_data.REFRACTORY_PERIOD.values[id_val],
                max_dosage=syringes_data.MAX_DOSAGE.values[id_val]
            )

        visits[str(id_visit)] = CreateDict(
            encounter_num=str(encounter_num),
            age=age,
            duration=duration,
            rea=str(rea),
            previous_visit=previous_visit,
            next_visit=next_visit,
            start_date=str(start_date),
            end_date=str(end_date),
            bio=bio_data,
            vital_parameters=vital_param,
            syringes=syringes,
            ORAL_OPIOID=str(ORAL_OPIOID),
            USED_MORPHINE=str(USED_MORPHINE),
            USED_OXYCODONE=str(USED_OXYCODONE),
            OPIOID_TO_DISCHARGE=str(OPIOID_TO_DISCHARGE),
            LS_INACTIVE=str(LS_INACTIVE),
            LS_ALONE=str(LS_ALONE),
            MH_ACS=str(MH_ACS),
            MH_PRIAPISM=str(MH_PRIAPISM),
            MH_AVN=str(MH_AVN),
            MH_ISCHEMIC_STROKE=str(MH_ISCHEMIC_STROKE),
            MH_LEG_ULCER=str(MH_LEG_ULCER),
            MH_HEART_FAILURE=str(MH_HEART_FAILURE),
            MH_PHTN=str(MH_PHTN),
            MH_RETINOPATHY=str(MH_RETINOPATHY),
            MH_NEPHROPATHY=str(MH_NEPHROPATHY),
            MH_DIALISIS=str(MH_DIALISIS),
            transfu_count=str(transfu_count)
        )
        id_visit += 1
    Patients[str(id_patient)] = CreateDict(
        patient_num=str(patient_num),
        sex=str(sex),
        birth_date=birth_date,
        visits=visits,
        death=str(death),
        death_date=str(death_date),
        ddn=ddn,
        baseline_HB=baseline_HB,
        genotype_SS=str(genotype_SS)
    )
    id_patient += 1

json_file_data = json.dumps(Patients)
json_file = open("json_file.json", "w")
json_file.write(json_file_data)
json_file.close()

print("\nJSON file created!")
