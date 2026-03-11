import numpy as np
import pandas as pd
from scipy.integrate import odeint
import os

def prion_kinetics(y,t,lam,gamma,k,mu):
    """Nucleated Polymerization Model ODE : 
    y[0] = M (Concentration of Normal PrPc)
    y[1] = I (Concentration of Misfolded PrPSc)
    """

    M,P = y 

    # dM/dt : Normal protein production minus natural clearance and disease conversion
    dM_dt = lam - (gamma*M) - (k*M*P)

    # dP/dt : Misfolded protein growth via conversion minus natural clearance
    dP_dt = (k*M*P) - (mu*P)

    return [dM_dt,dP_dt]

def generate_data(num_patients=500, output_dir='data/synthetic_cjd.csv'):
    print(f"Generating {num_patients} synthetic CJD patients...")

    lam = 1.0
    gamma = 0.1
    mu = 0.05

    data = []

    np.random.seed(42)
    
    for patient_id in range(1, num_patients+1):
        age = np.random.normal(65,10)

        genotype = np.random.choice(['MM', 'MV', 'VV'], p=[0.6,0.2,0.2])
        if genotype == 'MM':
            genetic_multiplier = 1.5
        elif genotype == 'MV':
            genetic_multiplier = 1.2
        elif genotype == 'VV':
            genetic_multiplier = 0.8

        age_factor = (age/65.0)
        patient_k = 0.15 * age_factor * genetic_multiplier

        M0 = lam/gamma
        P0 = np.random.uniform(0.01,0.05)
        y0 = [M0,P0]

        num_visits = np.random.randint(4,10)
        t_visits = np.sort(np.random.uniform(0,24,num_visits))
        t_visits[0] = 0.0

        solution = odeint(prion_kinetics,y0,t_visits,args=(lam,gamma,patient_k,mu))

        M_traj = solution[:,0] + np.random.normal(0,0.05,num_visits)
        P_traj = solution[:,1] + np.random.normal(0,0.05,num_visits)

        clinical_score = np.clip((P_traj/np.max(P_traj+1e-5))*100 + np.random.normal(0,2,num_visits), 0, 100)

        for i in range(num_visits):
            data.append({
                'Patient_ID': patient_id,
                'Visit_Month' : round(t_visits[i],2),
                'Age': age,
                'Genotype': genotype,
                'PrPC_Normal' : max(0,round(M_traj[i],4)),
                'PrPSc_Burden' : max(0,round(P_traj[i],4)),
                'Clinical_Score': clinical_score[i]
            })

            df = pd.DataFrame(data)
            df['Genotype_Num'] = df['Genotype'].map({'MM': 2, 'MV': 0, 'VV': 1})

            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            df.to_csv(output_dir, index=False)

            print(f"Generated data for patient {patient_id}")

if __name__ == "__main__":
    generate_data()