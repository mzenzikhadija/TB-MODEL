"""
Generate synthetic TB data

Most common symptoms occuring over 80% of the time (also halmark signs):
1. Cough for two weeks or more (productive)          *****************
2. Night sweats                                      *****************
3. Fever                                             *****************
4. Weight loss                                       *****************

Common symptoms occuring over 50% of the time
1. Chest pain                                         ****************
2. Malaise
3. Difficulty breathing

Signs and their prevalences:
Malnourished - 80%
Increased Respiratory rate - 50%
Reduced air entry/fluid filled lung - 50%             ***************


Risk Factors
1. Weakened immune system* (HIV) - 62% co morbidity   ****************
2. Having diabetes - twice as likely
3. Malnutrition (Low BMI) - 80%                       *****************
4. Recurrent infection of any kind - 80%
5. Substance abuse
6. Smoking                                            **************
7. Contact with TB
8. History of TB in the family - 80%                  *************

Gender distributions
Female - 50%
Male - 80%                                            **********

Age distribution
< 2 years - 10%
2 - 16 years - 25%
16+ - 65%

"""

# VERY IMPORTANT: You have to generate a lot of data (law of large numbers) so that they eventually converge to the probabilities 

from scipy.stats import bernoulli, halfnorm, norm
import csv
import random

print("Initializing ....")

# setting the random seed
random.seed(30)


# Making the strong assumption that the chance of having TB is 50/50
# (This is a mistake because the prevalence of TB is significatly lower than this)
p = 0.5

# How many data points are we creating (number of synthetic patients)
N_records = 10000


# TB positive case statuses
cases = bernoulli.rvs(p, size=N_records)


def generate_patient(status):
    """
    This function takes the status of a patient (1 = positive and 0 = negative)
    and returns the expected characteristics of this patient given the status
    """
    # sex is female with p=0.5 if the status is 0 and p=0.8 if status is 1
    # This means its more likely the patient is male if the status is 1
    sex = bernoulli.rvs(0.5) if status == 0 else bernoulli.rvs(0.8)

    # symptoms
    dry_cough = bernoulli.rvs(0.3) if status == 0 else bernoulli.rvs(0.6)
    productive_cough = bernoulli.rvs(0.3) if status == 0 else bernoulli.rvs(0.8)

    # NOTE: cough_duration should be enhanced with more stochasticity, you could be coughing for only 2 days but still be positive
    cough_duration = halfnorm.rvs(0, 2) if status == 0 else norm.rvs(14, 5) # Number of days coughing
    night_sweats = bernoulli.rvs(0.3) if status == 0 else bernoulli.rvs(0.8)
    fever = bernoulli.rvs(0.4) if status == 0 else bernoulli.rvs(0.8)
    weight_loss = bernoulli.rvs(0.3) if status == 0 else bernoulli.rvs(0.8)

    # dyspnoea is the same as difficulty breathing
    dyspnoea = bernoulli.rvs(0.2) if status == 0 else bernoulli.rvs(0.45)
    chest_pain = bernoulli.rvs(0.2) if status == 0 else bernoulli.rvs(0.45)


    # signs
    # if you dont have HIV, we assume the national average of 0.05
    hiv_positive = bernoulli.rvs(0.05) if status == 0 else bernoulli.rvs(0.62)
    mulnutrition = bernoulli.rvs(0.3) if status == 0 else bernoulli.rvs(0.8)

    # It is more likely to have TB if you are a smoker, and its more likely to be a smoker if you are male
    smoking = bernoulli.rvs(0.5) if sex == 1 else bernoulli.rvs(0.15) if status == 0 else bernoulli.rvs(0.8)
    family_tb = bernoulli.rvs(0.1) if status == 0 else bernoulli.rvs(0.8)

    return [sex, dry_cough, productive_cough, round(cough_duration), night_sweats, fever, weight_loss, dyspnoea, chest_pain, hiv_positive, mulnutrition, smoking, family_tb, status]


print(f"Generating {N_records} synthetic patients ...")
# loop through the statuses and create new patients based on the status
patients = [generate_patient(x) for x in cases]


print("Patient genertion completed. Writing to file")
with open('synthetic_tb_patients.csv', mode='w') as tb_file:
    tb_writer = csv.writer(tb_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    tb_writer.writerow(['sex', 'dry_cough', 'productive_cough', 'cough_duration', 'night_sweats', 'fever', 'weight_loss', 'dyspnoea', 'chest_pain', 'hiv_positive', 'mulnutrition', 'smoking', 'family_tb', 'status'])
    
    for patient in patients:
        tb_writer.writerow(patient)

print("Process completed")