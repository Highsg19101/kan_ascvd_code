import math
import numpy as np



def PREVENT(df,sex):
    age = df['AGE'].values # year
    tot = df['G1E_TOT_CHOL'].values #mg/dL
    hdl = df['G1E_HDL'].values #mg/dL
    sbp = df['G1E_BP_SYS'].values #mmHg
    dm = df['DM'].values 
    smk = df['SMK'].values
    htn = df['HTN_med'].values
    gfr = df['G1E_GFR'].values #ml/min/1.73m^2
    if sex == 0: # male
        log_odds = -3.500655+\
                    0.7099847 * (age - 55) /10 + \
                    0.1658663 * ((tot - hdl)*0.02586-3.5) -\
                    0.1144285 * (hdl * 0.02586 - 1.3) / 0.3 - \
                    0.2837212 * (np.where(sbp < 110, sbp, 110) - 110) / 20 + \
                    0.3239977 * (np.where(sbp > 110, sbp, 110) - 130) / 20 + \
                    0.7189597 * (dm) +\
                    0.3956973 * (smk) + \
                    0.3690075 * (np.where(gfr < 60, gfr, 60) - 60) / -15 + \
                    0.0203619 * (np.where(gfr > 60, gfr, 60) - 90) / -15 + \
                    0.2036522 * (htn) -\
                    0.0865581 * (0) - \
                    0.0322916 * (htn) * (np.where(sbp > 110, sbp, 110) - 130) /20 + \
                    0.114563 * (0) * ((tot - hdl) * 0.02586 - 3.5) - \
                    0.0300005 * (age - 55) / 10 * ((tot - hdl) * 0.02586 - 3.5) + \
                    0.0232747 * (age - 55) / 10 * (hdl * 0.02586 - 1.3) / 0.3 - \
                    0.0927024 * (age - 55) / 10 * (np.where(sbp > 110, sbp, 110) - 130) / 20 - \
                    0.2018525 * (age - 55) / 10 * (dm) - \
                    0.0970527 * (age - 55) /10 * (smk) - \
                    0.1217081 * (age - 55) /10 * (np.where(gfr < 60, gfr, 60) - 60) / -15

    elif sex == 1:
        log_odds = -3.819975+\
                    0.719883 * (age - 55)/10 +\
                    0.1176967 * ((tot - hdl) * 0.02586-3.5)-\
                    0.151185 * (hdl * 0.02586 - 1.3) /0.3 -\
                    0.0835358 * (np.where(sbp < 110, sbp, 110) - 110) /20+\
                    0.3592852 * (np.where(sbp > 110, sbp, 110) - 130) /20 +\
                    0.8348585 * (dm) +\
                    0.4831078 * (smk) +\
                    0.4864619 * (np.where(gfr < 60, gfr, 60) - 60) / -15 +\
                    0.0397779 * (np.where(gfr > 60, gfr, 60) - 90) / -15 +\
                    0.2265309 * (htn) -\
                    0.0592374 * (0) -\
                    0.0395762 * (htn) * (np.where(sbp > 110, sbp, 110) - 130) /20 +\
                    0.0844423 * (0) * ((tot - hdl) * 0.02586 - 3.5) -\
                    0.0567839 * (age - 55) /10 * ((tot - hdl) * 0.02586 - 3.5) +\
                    0.0325692 * (age - 55) /10 * (hdl * 0.02586 - 1.3)/0.3 -\
                    0.1035985 * (age - 55) /10 * (np.where(sbp > 110, sbp, 110) - 130) /20 -\
                    0.2417542 * (age - 55) /10 * (dm) -\
                    0.0791142 * (age - 55) /10 * (smk)-\
                    0.1671492 * (age - 55) /10 * (np.where(gfr < 60, gfr, 60) - 60) / -15
    else:
        raise ValueError("Sex type must be only 0 or 1.")
    risk = (np.exp(log_odds) / (1+np.exp(log_odds)))
    #print(log_odds, risk)
    return risk



def PCE_white(df, sex):
    age = df['AGE'].values # year
    tot = df['G1E_TOT_CHOL'].values #mg/dL
    hdl = df['G1E_HDL'].values #mg/dL
    sbp = df[['HTN_med','G1E_BP_SYS']].astype(float) #mmHg
    dm = df['DM'].values 
    smk = df['SMK'].values
    htn = df['HTN_med'].values
    gfr = df['G1E_GFR'].values #ml/min/1.73m^2

    ln_age = np.log(age)
    ln_age_square = np.log(age)**2
    ln_tot = np.log(tot)
    ln_age_tot = ln_age*ln_tot
    ln_hdl = np.log(hdl)
    ln_age_hdl = ln_age * ln_hdl

    if sex == 0:
        ln_age_v = 12.344*ln_age
        ln_tot_v = 11.853*ln_tot
        ln_age_tot_v = -2.664*ln_age*ln_tot
        ln_hdl_v = -7.990*ln_hdl
        ln_age_hdl_v = 1.769*ln_age*ln_hdl

        # Treated or Untreated
        sbp['G1E_BP_SYS'] = np.log(sbp['G1E_BP_SYS'].values)
        ln_sbp = sbp.values
        ln_sbp[ln_sbp[:,0]==0, 1] *= 1.764 # hyptertension medication = 0
        ln_sbp[ln_sbp[:,0]==1, 1] *= 1.797 # hyptertension medication = 1
        ln_sbp_v = ln_sbp[:,1]
        smk_v = 7.837*smk
        ln_age_smk_v = -1.795 * ln_age * smk
        dm_v = 0.658 * dm
        male = np.column_stack((
                           ln_age_v,
                           ln_tot_v,
                           ln_age_tot_v,
                           ln_hdl_v,
                           ln_age_hdl_v,
                           ln_sbp_v,
                           smk_v,
                           ln_age_smk_v,
                           dm))
        #print(np.sum(male, axis=1))
        risk = (1- (0.9144**np.exp(np.sum(male, axis=1) - 61.18)))
        return risk
    
    elif sex==1:
        ln_age_v = -29.799*ln_age
        ln_age_square_v = 4.884*ln_age_square
        ln_tot_v = 13.540*ln_tot
        ln_age_tot_v = -3.114*ln_age*ln_tot
        ln_hdl_v = -13.578*ln_hdl
        ln_age_hdl_v = 3.149*ln_age*ln_hdl

        # Treated or Untreated
        sbp['G1E_BP_SYS'] = np.log(sbp['G1E_BP_SYS'].values)
        ln_sbp = sbp.values
        ln_sbp[ln_sbp[:,0]==0, 1] *= 1.957 # hyptertension medication = 0
        ln_sbp[ln_sbp[:,0]==1, 1] *= 2.019 # hyptertension medication = 1
        ln_sbp_v = ln_sbp[:,1]
        smk_v = 7.574*smk
        ln_age_smk_v = -1.665 * ln_age * smk
        dm_v = 0.661 * dm
        female = np.column_stack((
                           ln_age_v,
                           ln_age_square_v,
                           ln_tot_v,
                           ln_age_tot_v,
                           ln_hdl_v,
                           ln_age_hdl_v,
                           ln_sbp_v,
                           smk_v,
                           ln_age_smk_v,
                           dm))
        #print(female)
        risk = (1- (0.9665**np.exp(np.sum(female, axis=1) - (-29.18))))
        #print(risk)
        return risk
    



def KRPM(df, sex):
    age = df['AGE'].values # year
    tot = df['G1E_TOT_CHOL'].values #mg/dL
    hdl = df['G1E_HDL'].values #mg/dL
    sbp = df[['HTN_med','G1E_BP_SYS']].astype(float) #mmHg
    dm = df['DM'].values 
    smk = df['SMK'].values
    htn = df['HTN_med'].values
    gfr = df['G1E_GFR'].values #ml/min/1.73m^2

    ln_age = np.log(age)
    ln_age_square = np.log(age)**2
    ln_tot = np.log(tot)
    ln_age_tot = ln_age*ln_tot
    ln_hdl = np.log(hdl)
    ln_age_hdl = ln_age * ln_hdl

    if sex == 0:
        ln_age_v = 9.362*ln_age
        ln_age_square_v = 2.425*ln_age_square
        ln_tot_v = 6.409*ln_tot
        ln_age_tot_v = -1.430*ln_age*ln_tot
        ln_hdl_v = -3.843*ln_hdl
        ln_age_hdl_v = 0.810*ln_age*ln_hdl

        
        # Treated or Untreated
        sbp['G1E_BP_SYS'] = np.log(sbp['G1E_BP_SYS'].values)
        ln_sbp = sbp.values.copy()
        #print(ln_sbp)
        ln_sbp[ln_sbp[:,0]==0, 1] *= 18.541 # hyptertension medication = 0
        ln_sbp[ln_sbp[:,0]==1, 1] *= 18.589 # hyptertension medication = 1
        ln_sbp_v = ln_sbp[:,1]
        #print('LnSBPvalues',ln_sbp_v)
        
        ln_age_sbp = sbp.values.copy()
        ln_age_sbp[:,1] = ln_age_sbp[:,1] * ln_age 
        ln_age_sbp[ln_age_sbp[:,0]==0, 1] *= -4.112 # hyptertension medication = 0
        ln_age_sbp[ln_age_sbp[:,0]==1, 1] *= -4.116 # hyptertension medication = 1
        ln_age_sbp_v = ln_age_sbp[:,1]
        #print(ln_age_sbp_v, ln_sbp_v)

        
        smk_v = 2.464*smk
        ln_age_smk_v = -0.503 * ln_age * smk
        dm_v = 0.410 * dm
        male = np.column_stack((
                           ln_age_v,
                           ln_age_square_v,
                           ln_tot_v,
                           ln_age_tot_v,
                           ln_hdl_v,
                           ln_age_hdl_v,
                           ln_sbp_v,
                           ln_age_sbp_v,
                           smk_v,
                           ln_age_smk_v,
                           dm))
        risk = (1- (0.96427**np.exp(np.sum(male, axis=1) - 87.556)))
        return risk
    
    
    elif sex == 1:
            ln_age_v = -9.519*ln_age
            ln_age_square_v = 3.417*ln_age_square
            ln_tot_v = 0.320*ln_tot
            #ln_age_tot_v = -1.430*ln_age*ln_tot
            ln_hdl_v = -0.476*ln_hdl
            #ln_age_hdl_v = 0.810*ln_age*ln_hdl


            # Treated or Untreated
            sbp['G1E_BP_SYS'] = np.log(sbp['G1E_BP_SYS'].values)
            ln_sbp = sbp.values.copy()

            ln_sbp[ln_sbp[:,0]==0, 1] *= 13.291 # hyptertension medication = 0
            ln_sbp[ln_sbp[:,0]==1, 1] *= 13.402 # hyptertension medication = 1
            ln_sbp_v = ln_sbp[:,1]

            ln_age_sbp = sbp.values.copy()
            ln_age_sbp[:,1] = ln_age_sbp[:,1] * ln_age 
            ln_age_sbp[ln_age_sbp[:,0]==0, 1] *= -2.876 # hyptertension medication = 0
            ln_age_sbp[ln_age_sbp[:,0]==1, 1] *= -2.889 # hyptertension medication = 1
            ln_age_sbp_v = ln_age_sbp[:,1]


            smk_v = 0.415*smk
            dm_v = 0.424 * dm
            female = np.column_stack((
                               ln_age_v,
                               ln_age_square_v,
                               ln_tot_v,
                               ln_hdl_v,
                               ln_sbp_v,
                               ln_age_sbp_v,
                               smk_v,
                               dm))
            risk = (1- (0.96963**np.exp(np.sum(female, axis=1) - 24.881)))
            #print(risk)
            return risk
        
        
def SCORE(df, sex):
    age = df['AGE'].values # year
    tot = df['G1E_TOT_CHOL'].values*0.0259 #mg/dL
    hdl = df['G1E_HDL'].values*0.0259 #mg/dL -> mmol/L 으로 변환 필요
    sbp = df['G1E_BP_SYS'].astype(float).values #mmHg
    smk = df['SMK'].values
    
    if sex ==0:
        cage = (age-60)/5
        cage_v = 0.3742*cage
        csmk_v = 0.6012*smk
        csbp = (sbp-120)/20
        csbp_v = 0.2777*csbp
        ctot = (tot-6)/1
        ctot_v = 0.1458*ctot
        chdl = ((hdl-1.3)/0.5)
        chdl_v = -0.2698*chdl
        csmkage_v = -0.0755*(cage*smk)
        #print(csbp,cage)
        csbpage_v = -0.0255*(csbp*cage)
        ctotage_v = -0.0281*(ctot*cage)
        chdlage_v = 0.0426*(chdl*cage)
        
        male = np.column_stack((
                               cage_v,
                               csmk_v,
                               csbp_v,
                               ctot_v,
                               chdl_v,
                               csmkage_v,
                               csbpage_v,
                               ctotage_v,
                               chdlage_v
                              ))
        male_sum = np.sum(male, axis=1)
        risk_estim = 1-0.9605**np.exp(male_sum)
        risk_row = 1-np.exp(-np.exp(-0.5699+(0.7476*np.log(-np.log(1-risk_estim)))))
        return risk_row
        
    elif sex == 1:
        cage = (age-60)/5
        cage_v = 0.4648*cage
        csmk_v = 0.7744*smk
        csbp = (sbp-120)/20
        csbp_v = 0.3131*csbp
        ctot = (tot-6)/1
        ctot_v = 0.1002*ctot
        chdl = ((hdl-1.3)/0.5)
        chdl_v = -0.2606*chdl
        csmkage_v = -0.1088*(cage*smk)
        #print(csbp,cage)
        csbpage_v = -0.0277*(csbp*cage)
        ctotage_v = -0.0226*(ctot*cage)
        chdlage_v = 0.0613*(chdl*cage)
        
        female = np.column_stack((
                               cage_v,
                               csmk_v,
                               csbp_v,
                               ctot_v,
                               chdl_v,
                               csmkage_v,
                               csbpage_v,
                               ctotage_v,
                               chdlage_v
                              ))
        female_sum = np.sum(female, axis=1)
        risk_estim = 1-0.9776**np.exp(female_sum)
        risk_row = 1-np.exp(-np.exp(-0.7380+(0.7019*np.log(-np.log(1-risk_estim)))))
        return risk_row

    