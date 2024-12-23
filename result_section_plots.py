# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:18:27 2024

@author: tgilh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:56:23 2024

@author: tgilh
"""
#%% Choose which wind speed to evaulate (only 1 can be true)
use_v_y             = True
use_v_y_estimation  = False
use_v_los           = False
#%% IMPORTS
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import scipy.stats as stats
# import statsmodels.api as sm
import seaborn as sns
# import pylab as py 
sns.set(style="whitegrid")  # You can choose any style that fits your aesthetic needs: darkgrid, whitegrid, dark, white, ticks
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import warnings
from scipy.stats import linregress
# Suppress repeated RuntimeWarnings and show each warning only once
warnings.simplefilter("once", RuntimeWarning)
# Suppress only the specific RuntimeWarning related to log operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")

#%% PATH AND FILE READING
# path = 'C:/Users/tgilh/hublidardatabase/files/df_huli.pkl'
# with open(path, 'rb') as file:
#    dict_lidar = pickle.load(file)  
  
# df_original=dict_lidar['dtu_10mw_wsp_8.0_seed_927_ae_0.138']['config_1']

# df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_u.pkl")
# df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_u_config1.pkl")
# df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_u_config2.pkl")

df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_u_config2_11_4_m_s.pkl")
# df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_u_config2_18_m_s.pkl")
# df_v_y_extracted = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_points_wind_ramp.pkl")

# dict_lidar = pd.read_pickle("C:/Users/tgilh/hublidardatabase/files/df_wind_ramp.pkl")
# df_v_y_extracted=dict_lidar['dtu_10mw_wsp_11.4_seed_927_ae_0.181_wind_ramp']['config_1']



# # Concatenate the dataframe with itself
# # Determine the time period (difference between max and min time)
# time_shift = df_v_y_extracted['Time'].max() - df_v_y_extracted['Time'].min()

# # # Create a shifted copy of the dataframe
# df_shifted = df_v_y_extracted.copy()
# df_shifted2 = df_v_y_extracted.copy()

# df_shifted['Time'] += time_shift + 1  # Shift by the full period plus 1 to avoid overlap
# df_shifted2['Time'] += 2*time_shift + 1  # Shift by the full period plus 1 to avoid overlap

# # Concatenate the original dataframe with the shifted one
# df_concatenated = pd.concat([df_v_y_extracted, df_shifted,df_shifted2], ignore_index=True)
# df_v_y_extracted = df_concatenated.copy()


#%% RENAMING COLUMNS ETC.
# df_original.rename(columns={
#     'HuLi_Xg': 'x',
#     'HuLi_Yg': 'y',
#     'HuLi_Zg': 'z',
#     'HuLi_V_LOS_wgh': 'v',
#     'HuLi_V_LOS_nom': 'vnom',
#     'hub1_pos_x':'xhub',
#     'hub1_pos_y':'yhub',
#     'hub1_pos_z':'zhub',
    
# }, inplace=True)

df_v_y_extracted.rename(columns={
    'HuLi_Xg': 'x',
    'HuLi_Yg': 'y',
    'HuLi_Zg': 'z',
    'Interpolated_Wind_Speed': 'v_interpolated',
    'HuLi_V_LOS_wgh_scaled': 'v_y_estimated',
    'HuLi_V_LOS_wgh':   'v_los'
    ''
    
}, inplace=True)
df_v_y_extracted = df_v_y_extracted.drop(['time_tbox',  'HuLi_Yt', 'x_user', 'xt_user', 'y_user', 'z_user', 'HuLi_V_LOS_nom_scaled',
       'HuLi_V_LOS_nom', 'x_near', 'y_near', 'z_near', 'u_Nearest',
       'v_Nearest', 'w_Nearest', 'HuLi_Yt_near', 'HuLi_Xg_near',
       'HuLi_Zg_near'], axis=1)

chosen_v = 'v not chosen'

def chooseWS_Name():
    
    chosen_v_local = 'v not chosen'

    if use_v_y == True:
        chosen_v_local = 'true u'
    if use_v_y_estimation == True:
        chosen_v_local = 'u_estimated'
    if use_v_los == True:
        chosen_v_local = 'v_los'
        
    print(f'chosen wind speed: {chosen_v_local}')

    return chosen_v_local

chosen_v = chooseWS_Name()
        

def chooseWS(ws,name):    
    
    if ws== True:
        df_v_y_extracted.rename(columns={
            f'{name}': 'v',         
        }, inplace=True)
        return df_v_y_extracted
    else:
        return df_v_y_extracted

df_v_y_extracted= chooseWS(use_v_y,                'v_interpolated')
df_v_y_extracted= chooseWS(use_v_y_estimation,     'v_y_estimated')
df_v_y_extracted= chooseWS(use_v_los,              'v_los')






#%% CONSTRAIN VOLUME OF MEASUREMENTS TO ACTUAL TURBULENCE BOX
def invertAxes(df):
     df['y'] = -df['y']
     df['z'] = -df['z']
     return df

# Inverting axes for positive heights and y values
df_v_y_extracted = invertAxes(df_v_y_extracted)

def constrainVolumeToTurbBox(df):
     valid_indices = (df['z'] > 0) & (df['v'] > 0) & (df['z'] <= 400) & (df['x'] <= 200) & (df['x'] >= -200)
     return df[valid_indices].reset_index(drop=True)

# Update your DataFrames by assigning the returned value
df_v_y_extracted = constrainVolumeToTurbBox(df_v_y_extracted)

# Box Size
x_range = [-200,200]
y_range = [45,500]
z_range = [int(df_v_y_extracted['z'].min()),int(df_v_y_extracted['z'].max())] #box height
v_range = [df_v_y_extracted['v'].min(),df_v_y_extracted['v'].max()]

#%%definitions for optimizing and checking algorithm
advection_df = pd.DataFrame({    
    'Power_Curved'  : [],
    'Power_Linear'  : [],
    'Log'           : [],
    'Poly'          : []
})

rmse_df= pd.DataFrame({    
    'Power_Curved'  : [],
    'Power_Linear'  : [],
    'Log'           : [],
    'Poly'          : []
})

params_df = pd.DataFrame({
    'alpha_curved': [],
    'alpha_linear': [],
    'u_star': [],
    'z0': []
})

#point counter
points_df = pd.DataFrame({
    'time'       :[],
    'distance'   :[],
    'grouped'    :[],
    'merged'     :[]
    })

storage_df = pd.DataFrame()
#%% Post processing functions
def getSpeedAtZ_ref():
    # Convert 'z_avg' to a numeric type if it's categorical
    df_grouped['z_avg'] = pd.to_numeric(df_grouped['z_avg'], errors='coerce')
    
    # Apply the height mask
    height_mask = (df_grouped['z_avg'] <= z_ref+bin_size) & (df_grouped['z_avg'] >= z_ref-bin_size)
    df_height_masked = df_grouped[height_mask].copy()  # Use .copy() here to avoid warnings
    df_height_masked.reset_index(drop=True, inplace=True)
    
    lowerHeightSpeed = df_height_masked['u'][0]
    lowerHeight = df_height_masked['z_avg'][0]
    upperHeightSpeed = df_height_masked['u'][1]
    upperHeight = df_height_masked['z_avg'][1]
    
    v_ref_binned = linInterp(lowerHeight, upperHeight, lowerHeightSpeed, upperHeightSpeed, z_ref)

    print(f'ws @{lowerHeight}m = {lowerHeightSpeed:.2f}')
    print(f'v_ref_binned = {v_ref_binned:.2f} (Interpolated from bins)' )               
    print(f'ws @{upperHeight}m = {upperHeightSpeed:.2f}')
    print()
       
    return v_ref_binned, lowerHeightSpeed,lowerHeight,upperHeightSpeed,upperHeight

# def advec_updater(u_ref_name,function,parameters, list_name):
#     u_ref_name = function(*parameters)
#     list_name.append(u_ref_name)
#     return list_name, u_ref_name  

def calculateMeanAdvectionSpeed(function,parameters):
    return function(*parameters)

def calculateMeanAdvectionSpeeds():
    pow_ws_curv_mean    = calculateMeanAdvectionSpeed(power_law, [z_ref, alpha_curved,V_ref])
    # pow_ws_curv_mean    = power_law(z_ref, alpha_curved,V_ref)
    # pow_ws_lin_mean     = power_law(z_ref, alpha_linear,V_ref)
    pow_ws_lin_mean     = calculateMeanAdvectionSpeed(power_law, [z_ref, alpha_linear,V_ref])
    log_ws_mean         = calculateMeanAdvectionSpeed(log_wind_profile,[z_ref,*log_parameters])
    poly_ws_mean        = calculateMeanAdvectionSpeed(np.polyval,[poly_coeffs,z_ref])
    u_adv_mean          = np.mean([pow_ws_curv_mean,log_ws_mean,pow_ws_lin_mean,poly_ws_mean])
    return u_adv_mean 

def appendToDf(function,parameters, col_name,rmse):
    advection_df.loc[len(advection_df), f'{col_name}'] = calculateMeanAdvectionSpeed(function,parameters)
    rmse_df.loc[len(rmse_df), f'{col_name}'] = rmse    
    # binned_speeds_df.loc[len(binned_speeds_df), 'lowerHeightSpeed'] = getSpeedAtZ_ref()[0]
    # binned_speeds_df.loc[len(binned_speeds_df), 'lowerHeight'] = getSpeedAtZ_ref()[1]
    # binned_speeds_df.loc[len(binned_speeds_df), 'upperHeightSpeed'] = getSpeedAtZ_ref()[2]
    # binned_speeds_df.loc[len(binned_speeds_df), 'upperHeight'] = getSpeedAtZ_ref()[3]
    return advection_df, rmse_df

def pointsCounter(col_name,df):
    points_df.loc[len(points_df),f'{col_name}']=len(df)
    return points_df

def updatePointsCounter():
    pointsCounter('time',df_time_masked)
    pointsCounter('distance',df_distance_masked)
    pointsCounter('grouped',df_grouped)
    pointsCounter('merged',df_merged)
    return points_df

 # Append to parameters DataFrame
def appendParams(alpha_curved=None, alpha_linear=None, u_star=None, z0=None):
    params_df.loc[len(params_df), 'alpha_curved'] = alpha_curved
    params_df.loc[len(params_df), 'alpha_linear'] = alpha_linear
    params_df.loc[len(params_df), 'u_star'] = u_star
    params_df.loc[len(params_df), 'z0'] = z0
    return params_df

def updateAllDfs():
    appendToDf(power_law,[z_ref,alpha_curved,V_ref],'Power_Curved',rmse_rotor_plane_curved)
    appendToDf(power_law,[z_ref,alpha_linear,V_ref],'Power_Linear',rmse_rotor_plane_linear)
    appendToDf(log_wind_profile,[z_ref,*log_parameters],'Log',rmse_rotor_plane_log)
    appendToDf(np.polyval,[poly_coeffs,z_ref],'Poly',rmse_rotor_plane_poly)
    appendParams(alpha_curved=alpha_curved, alpha_linear=alpha_linear, u_star=log_parameters[0], z0=log_parameters[1])
    return advection_df, rmse_df, params_df

def cleanDf():
    # For advection_df
    def cleaner(df):
        for col in df.columns:
            df[col] = df[col].dropna().reset_index(drop=True) #remove cells in NaN columns 
        df.dropna(inplace=True) # Remove rows with any NaNs in the DataFrame
        
    cleaner(advection_df)
    cleaner(rmse_df)
    cleaner(points_df)
    cleaner(params_df)
    # cleaner(binned_speeds_df)
    
def truePowerLaw():
    #height list for rmse computation (and plotting?)
    heights = np.linspace(z_min,z_max,z_range[1])
    truePowerLawSpeeds = power_law(heights, trueAlpha,u_adv_true)
    
    # truePowerLawSpeeds = trueV_ref*(heights / z_ref)**trueAlpha
    return truePowerLawSpeeds,heights

def linInterp(x1,x2,y1,y2,x):
    y = y1+(x-x1)*((y2-y1)/(x2-x1))
    return y



#%%RMSE FUNCTION
def compute_rmse_true_power_law(fitted_speeds,binned_heights,binned_speeds):
    # true_power_law_binned_heights = np.linspace(heights[0],heights[-1],len(binned_speeds))
    heights = np.linspace(z_min,z_max,z_range[1])
    true_power_law_binned_speeds = power_law(binned_heights,trueAlpha,u_adv_true)
    
    # Define the bounds of the rotor plane with a buffer
    lower_bound = rotor_lowest_height - bin_size
    upper_bound = rotor_highest_height + bin_size
    
    # Create a mask for the range of heights in the rotor plane
    rotor_mask = (heights >= lower_bound) & (heights <= upper_bound)
    
    # Apply the mask to select the appropriate fitted and true speeds
    truePowerLawSpeeds = power_law(heights, trueAlpha,u_adv_true)
    fitted_subset = fitted_speeds[rotor_mask]
    true_subset = truePowerLawSpeeds[rotor_mask]
    
    # Compute the RMSE for the rotor plane
    rmse_rotor_plane = np.sqrt(mean_squared_error(fitted_subset, true_subset))    
    rmse_full_height = np.sqrt(mean_squared_error(fitted_speeds, truePowerLawSpeeds))
    rmse_binned_speeds = np.sqrt(mean_squared_error(binned_speeds, true_power_law_binned_speeds))
    return rmse_rotor_plane, rmse_full_height, rmse_binned_speeds

# Relative RMSE
v_range_post = v_range[1]-v_range[0]
def rmse_rel(rmse,fit_name):
    rmse_rel = (rmse / v_range_post) * 100
    print(f'Relative RMSE - {fit_name} full height: {rmse_rel:.1f}%')
    return rmse_rel

# def compute_rmse_binned(binned_speeds, fitted_speeds):    
#     return np.sqrt(mean_squared_error(binned_speeds, fitted_speeds))
#%% POWER LAW FUNCTIONS
 # Define the power law model to fit only the shear exponent (alpha)
def power_law(z, alpha, v_ref):
    return v_ref*(z / z_ref)**alpha

def power_curved_fit(z, v, alpha, v_ref):
    """
    Fit the power law model using a fixed reference height and wind speed.
    Args:
        z (np.ndarray): Array of heights (z values).
        v (np.ndarray): Array of wind speeds at corresponding heights.
    Returns:
        power_curved_fitted_speeds (np.ndarray): fitted values
        rmse (float): computed with compute_rmse
    """
    alpha_guess = [alpha]
    # Use lambda to optimize only alpha
    params, _ = curve_fit(lambda z, alpha: power_law(z, alpha,v_ref), z, v, p0=alpha_guess)
    alpha = params[0]
    
    # Calculate the fitted wind speeds using the fitted alpha
    heights = np.linspace(z_min,z_max,z_range[1])
    power_curved_fitted_speeds = power_law(heights, alpha,v_ref)
    
    # Calculate RMSE
    rmse_rotor_plane, rmse_full_height, rmse_binned_speeds = compute_rmse_true_power_law(power_curved_fitted_speeds,z,v)  
    
    return power_curved_fitted_speeds, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, alpha

def power_fit_linearized(z, v, alpha, v_ref):
    """
    Fits a power law model to the given data by linearizing it.

    Parameters:
    - z (array-like): Array of heights.
    - v (array-like): Array of wind speeds at corresponding heights.
    - z_ref (float): Reference height.
    - v_ref (float): Wind speed at the reference height.

    Returns:
    - alpha (float): The fitted shear exponent.
    - v_fitted (np.ndarray): Fitted wind speeds based on the model.
    - rmse (float): Root Mean Square Error of the fit.
    """
    heights = np.linspace(z_min,z_max,z_range[1])
    # Linearize the data
    x = np.log(z / z_ref)      # x = log(z / z_ref)
    y = np.log(v / v_ref)      # y = log(v / v_ref)

    # Perform linear regression to get slope (alpha) and intercept
    slope, intercept, _, _, _ = linregress(x, y)

    # Convert slope back to alpha and calculate fitted speeds
    alpha = slope   
    v_fitted = v_ref * (heights / z_ref) ** alpha

    # Compute RMSE
    rmse_rotor_plane, rmse_full_height, rmse_binned_speeds = compute_rmse_true_power_law(v_fitted,z,v)

    return v_fitted, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, alpha
#%% LOG FIT FUNCTIONS
def log_wind_profile(z, u_star, z0):
    kappa = 0.4
    return (u_star / kappa) * np.log(z / z0)

def logarithmic_fit(z, v):
    heights = np.linspace(z_min,z_max,z_range[1])
    params, _ = curve_fit(log_wind_profile, z, v, p0=[0.5, 0.1])
    u_star, z0 = params
    log_fitted_speeds = log_wind_profile(heights, u_star, z0)
    rmse_rotor_plane, rmse_full_height, rmse_binned_speeds = compute_rmse_true_power_law(log_fitted_speeds,z,v)
    # rmse = compute_rmse_binned(v, log_fitted_speeds)
    return log_fitted_speeds,rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, params
#%% POLYNOMIAL FIT FUNCTIONS
def polynomial_fit(z, v):
    heights = np.linspace(z_min,z_max,z_range[1])
    poly_coeffs = np.polyfit(z, v, poly_degree)
    poly_fitted_speeds = np.polyval(poly_coeffs, heights)
    rmse_rotor_plane, rmse_full_height, rmse_binned_speeds = compute_rmse_true_power_law(poly_fitted_speeds,z,v)
    # rmse = compute_rmse_binned(v, poly_fitted_speeds)
    return poly_fitted_speeds,rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, poly_coeffs
#%% PLOT FUNCTIONS
def plot_vertical(df_wind_speed1, df_height1, label1,                  
                  title, rmse=0, 
                  add_val1=0, add_val_label1='none', 
                  add_val2=0, add_val_label2='none',
                  add_val3=0, add_val_label3='none'):
    """
    Plots vertical wind profile with two sets of wind speeds and heights.
    Optional: includes RMSE and additional values like counts or points if provided.
    """
    plt.figure()
    plt.scatter(df_wind_speed1, df_height1, s=5, color='red', label=label1)
    if rmse != 0:
        plt.plot([], [], ' ', label=f'RMSE={rmse:.4f}')    
    if add_val_label1 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label1} = {add_val1}')
    if add_val_label2 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label2} = {add_val2}')    
    if add_val_label3 != 'none':                
        plt.plot([], [], ' ', label=f"{add_val_label3} = {add_val3}")
    plt.xlim(0, 20)
    plt.ylim(-10,500)
    plt.xlabel('wind speed [m/s]')
    plt.ylabel('z position [m]')    
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()

def plot_vertical_specific(df_x,df_y, marker):
    if marker == True:
        plot_vertical(df_x,df_y, 'binned speeds',               
                  f'Binned speeds. t={t}', 
                  add_val1=len(df_merged["u"]), add_val_label1='amount of datapoints',
                  add_val2=bin_size,add_val_label2='Bin Size [m]')

def plot_vertical_true_power(df_x,df_y, marker):
    """
    Plots vertical wind profile with two sets of wind speeds and heights.
    Optional: includes RMSE and additional values like counts or points if provided.
    """
    
    trueSpeeds, trueHeights = truePowerLaw()
    rmse = rmse_full_height_curved
    relative_rmse = rmse_rel(rmse,'Power_Curved')
    alpha_estimate = params_df['alpha_curved'].mean()
    advection_df['average_advec_log_pol'] = advection_df[['Log', 'Poly']].mean(axis=1)
    advection_df['average_advec_log_pol_power'] = advection_df[['Log', 'Poly','Power_Curved','Power_Linear']].mean(axis=1)
    U_estimate_log_pol = advection_df['average_advec_log_pol'].mean()
    U_estimate_log_pol_power = advection_df['average_advec_log_pol_power'].mean()
    

        
    plt.figure()    
    plt.scatter(trueSpeeds,     trueHeights, color = 'blue',s = 2)
    plt.plot(trueSpeeds+50,     trueHeights, color = 'blue', label = 'True Power Law')
    plt.scatter(df_x, df_y, color='red', label='filtered speeds', s = 8,alpha=1)    
    plt.plot([], [], ' ', label=f'Estimated Alpha={alpha_estimate:.2f}')    
    plt.plot([], [], ' ', label=f'Advection Speed = {u_adv_true:.2f}')    
    # plt.plot([], [], ' ', label=f'Estimated V_ref_log_pol= {U_estimate_log_pol:.2f}')    
    plt.plot([], [], ' ', label=f'Estimated U= {U_estimate_log_pol_power:.2f}')    
    plt.plot([], [], ' ', label=f'RMSE full height= {rmse:.2f}')
    plt.plot([], [], ' ', label=f'RMSE rotor range= {rmse_rotor_plane_curved:.2f}')
    plt.plot([], [], ' ', label=f'RMSE Relative= {relative_rmse:.2f}')
    plt.plot([], [], ' ', label=f'RMSE filtered speeds= {rmse_binned_speeds_curved:.2f}')
    plt.plot([0,20],[rotor_lowest_height]*2,label = 'Rotor range', color = 'grey', linestyle = ':')
    plt.plot([0,20],[rotor_highest_height]*2, color = 'grey', linestyle = ':')



    plt.xlim(0, 30)
    plt.ylim(-10,400)
    plt.xlabel('wind speed [m/s]')
    plt.ylabel('z position [m]')    
    plt.legend(loc='lower right')
    plt.title(f'{chosen_v}. t={t:.0f}s')
    plt.show()    

def plot_vertical_fit(df_wind_speed1, df_height1, label1,
                  df_wind_speed2, df_height2, label2,
                  title, rmse=0, 
                  add_val1=0, add_val_label1='none', 
                  add_val2=0, add_val_label2='none',
                  add_val3=0,add_val_label3='none'):
    """
    Plots vertical wind profile with two sets of wind speeds and heights.
    Optional: includes RMSE and additional values like counts or points if provided.
    """
    
    trueSpeeds, trueHeights = truePowerLaw()
        
    plt.figure()
    plt.scatter(df_wind_speed2, df_height2, color='black', label=label2, s = 1)    
    plt.scatter(df_wind_speed1, df_height1, color='red', label=label1,s=1)
    plt.scatter(trueSpeeds,     trueHeights, color = 'blue', label = 'True Power Law', s =5)
    plt.plot([0,20],[rotor_lowest_height]*2,label = 'Rotor Bottom', color = 'grey', linestyle = ':')
    plt.plot([0,20],[rotor_highest_height]*2,label = 'Rotor Top', color = 'grey', linestyle = ':')
        
    if rmse != 0:
        plt.plot([], [], ' ', label=f'RMSE={rmse:.4f}')    
    if add_val_label1 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label1} = {add_val1}')
    if add_val_label2 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label2} = {add_val2}')    
    if add_val_label3 != 'none':                
        plt.plot([], [], ' ', label=f"{add_val_label3} = {add_val3}")
    plt.xlim(0, 20)
    plt.ylim(-10,500)
    plt.xlabel('wind speed [m/s]')
    plt.ylabel('z position [m]')    
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()    

def plot_estimation_box(df_y_1, df_z_1, label1,
                        df_y_2, df_z_2, label2, 
                        title,
                        xval = 'x', yval = 'y',
                        xlim=[-50,600], ylim=[-10,500], 
                        add_val1=0, add_val_label1='none', 
                        add_val2=0, add_val_label2='none',
                        add_val3=0, add_val_label3='none'):
    """
    Plots estimation box, with two datasets eg. shifted and original positions. 
    Optional: includes additional values like counts or points if provided.
    """
    plt.figure()
    plt.scatter(df_y_2, df_z_2, color='blue',s=0.05, label=label2)
    plt.scatter(df_y_1, df_z_1, color='red',s=10, label=label1)    
    if add_val_label1 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label1} = {add_val1}')    
    if add_val_label2 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label2} = {add_val2}')
    if add_val_label3 != 'none':
        plt.plot([], [], ' ', label=f'{add_val_label3} = {add_val3}')    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(f'{xval} position [m]')
    plt.ylabel(f'{yval} position [m]')    
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
    
def plot_estimation_box_y_vs_z(marker):
    if marker == True:
        plot_estimation_box(df_merged['y_new'], df_merged['z'], 'binned speeds',
                            df_time_masked['y'], df_time_masked['z'], 'original position of measurements', 
                            title = 'Estimation Box. Advected Measurements',
                            add_val1=timescale, add_val_label1='timescale',
                            add_val2=len(df_time_masked), add_val_label2 = 'amount of datapoints',
                            add_val3 = bin_size, add_val_label3 = 'Bin Size [m]',
                            xval='y', yval='z')       
    
def plot_estimation_box_x_vs_y(marker):
    if marker == True:
        plot_estimation_box(df_merged['x'], df_merged['y_new'], 'advected measurement positions binned in estimation box',
                    df_active['x'], df_active['y'], 'original position of measurements', 
                    title = 'Estimation Box: x and y values of measurements',
                    add_val1=timescale, add_val_label1='timescale',
                    add_val2=len(df_time_masked), add_val_label2 = 'amount of datapoints',
                    add_val3 = bin_size, add_val_label3 = 'Bin Size [m]',
                    xval='x', yval='y',
                    xlim=[-300,300], ylim=[-10,600], )  

def plot_estimation_box_x_vs_z(marker):
    if marker == True:
        plot_estimation_box(df_merged['x'], df_merged['z'], 'advected measurement positions binned in estimation box',
                        df_active['x'], df_active['z'], 'original position of measurements', 
                        title = 'Estimation Box: x and y values of measurements',
                        add_val1=timescale, add_val_label1='timescale',
                        add_val2=len(df_time_masked), add_val_label2 = 'amount of datapoints',
                        add_val3 = bin_size, add_val_label3 = 'Bin Size [m]',
                        xval='x', yval='z',
                        xlim=[-300,300], ylim=[-100,500], )  

def plot_estimation_box_no_bins(marker):
    if marker == True:
        plot_estimation_box(df_distance_masked['y_new'], df_distance_masked['z'], 'binned speeds',
                        df_time_masked['y'], df_time_masked['z'], 'original position of measurements', 
                        title = 'Estimation Box. Advected Measurements',
                        add_val1=timescale, add_val_label1='timescale',
                        add_val2=len(df_time_masked), add_val_label2 = 'amount of datapoints',
                        add_val3 = bin_size, add_val_label3 = 'Bin Size [m]',
                        xval='y', yval='z')    

# plot counts of each bin 
# Assuming df_grouped has a 'count' column and 'z_avg' represents the midpoint of each bin
def plot_counts_barplot(df):
    plt.figure(figsize=(10, 6))  # Optional: To make the plot larger
    plt.bar(df['z_avg'], df['count'], width=5)  # Use z_avg as the x-axis, count as y-axis
    # Add labels and title
    plt.xlabel('Height Bin Midpoints (z_avg)')
    plt.ylabel('Number of Measurements (count)')
    plt.title('Distribution of Measurements in Each Height Bin')    
    # Optionally rotate x-axis labels if they overlap
    plt.xticks(rotation=45)    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_amount_of_points_vs_time():
    time_values = np.linspace(t_fill_up,t_cut_off,len(points_df['time']))

    plt.figure()
    plt.scatter(time_values,points_df['time'], label = 'df_time_masked')
    # plt.scatter(time_arr,points_distance,label = 'df_distance_masked')
    # plt.scatter(time_arr,points_grouped,label = 'df_grouped')
    # plt.scatter(time_arr,points_merged,label = 'df_merged')
    plt.plot([], [], ' ', label=f'timescale={timescale}')   
    plt.title('amount of points over time')
    plt.legend()
    plt.xlabel('seconds')
    plt.ylabel('points in dataframe')
    plt.show()

def calcMeanVal(df):
    meanValue = np.mean([df['Power_Curved'].mean(), df['Power_Linear'].mean(), df['Log'].mean(), df['Poly'].mean()])
    meanLastVal = np.mean([df['Power_Curved'].iloc[-1],df['Power_Linear'].iloc[-1],df['Log'].iloc[-1],df['Poly'].iloc[-1]])
    bestValPowerCurved = df['Power_Curved'].iloc[-1]
    bestValPowerLinear = df['Power_Linear'].iloc[-1]
    bestValLog = df['Log'].iloc[-1]
    bestValPoly = df['Poly'].iloc[-1]

    return meanValue, meanLastVal, bestValPowerCurved, bestValPowerLinear, bestValLog, bestValPoly



def plot_mean_value_vs_time(df,name,ylim, advec = False, mean = False, legend= False, legendPos = 'lower right'):
    # Generate a list with `meanValue` repeated for the length of `time_values`
    time_values = np.linspace(t_fill_up, t_cut_off, len(df['Log']))
    meanValue = calcMeanVal(df)[0]
    meanValueList = [meanValue] * len(time_values)
    ExpectedValList = [u_adv_true] * len(time_values)
    # time_values_for_binned_speeds = np.linspace(t_fill_up, t_cut_off, len(binned_speeds_df))
    # meanLowerAdvecBinnedList = [binned_speeds_df['lowerHeightSpeed'].mean()]*len(time_values_for_binned_speeds)
    # meanUpperAdvecBinnedList = [binned_speeds_df['upperHeightSpeed'].mean()]*len(time_values_for_binned_speeds)
    
    plt.figure()
    plt.plot(time_values,df['Power_Curved'],label='power (curve fit)')
    plt.plot(time_values,df['Power_Linear'],label='power (linearized fit)')
    plt.plot(time_values,df['Log'],label='log')
    plt.plot(time_values,df['Poly'],label='poly')
    plt.plot(time_values,meanValueList,label = f'{name} = {meanValue:.2}', linestyle = '--')
    plt.plot(time_values, ExpectedValList, label=f'expected u_adv={u_adv_true:.1f}',color='black')  
    # if advec == True:
    #     plt.plot(time_values_for_binned_speeds, binned_speeds_df['lowerHeightSpeed'], 
    #              label = f"v_bin, z={binned_speeds_df['lowerHeight'][0]}")
    #     plt.plot(time_values_for_binned_speeds, binned_speeds_df['upperHeightSpeed'], 
    #              label = f"v_bin, z={binned_speeds_df['upperHeight'][0]}")
        
    # if mean == True:
    #     plt.plot(time_values_for_binned_speeds, meanLowerAdvecBinnedList, 
    #              label = f"v_bin, z={binned_speeds_df['lowerHeight'][0]}")
    #     plt.plot(time_values_for_binned_speeds, meanUpperAdvecBinnedList, 
    #              label = f"v_bin, z={binned_speeds_df['upperHeight'][0]}")
        
        
    plt.ylim(ylim)
    plt.xlabel('seconds')
    plt.title(f'{name} over time')
    if legend == True:
        plt.legend(loc=legendPos)
    plt.show()

def calcMeanAlpha():
    meanAlpha = np.mean([params_df['alpha_curved'].mean(), params_df['alpha_linear'].mean()])
    bestAlphaCurved = params_df['alpha_curved'].iloc[-1]
    bestAlphaLinear = params_df['alpha_linear'].iloc[-1]
    meanLastAlpha = np.mean([bestAlphaCurved,bestAlphaLinear])

    return meanAlpha, bestAlphaCurved, bestAlphaLinear, meanLastAlpha
    
def plot_parameters_vs_time(ylim):
    time_values = np.linspace(t_fill_up,t_cut_off,len(params_df))
    meanAlpha = calcMeanAlpha()[0]
    meanAlphaList = [meanAlpha] * len(time_values)
    expectedAlpha = 0.2
    expectedAlphaList = [meanAlpha] * len(time_values)
    
    plt.figure()
    plt.plot(time_values,params_df['alpha_curved'],label='Alpha: power (curve fit)')
    # plt.plot(time_values,params_df['alpha_linear'],label='Alpha: power (linearized fit)')
    # plt.plot(time_values,params_df['u_star'],label='Log fit: Friction Velocity')
    # plt.plot(time_values,meanAlphaList,label = f'mean alpha = {meanAlpha:.3}', linestyle = '--', color = 'red')
    plt.plot(time_values,expectedAlphaList,label = f'Expected Alpha = {expectedAlpha:.3}', linestyle = ':',color= 'black')
    plt.ylim(ylim)
    plt.xlabel('seconds')
    plt.title('Alpha over time')
    plt.legend()
    plt.show()    
    
def plot_z0_vs_time(ylim):
    time_values = np.linspace(t_fill_up,t_cut_off,len(params_df))
    
    plt.figure()
    plt.plot(time_values[0:-2],params_df['z0'][0:-2],label='Log fit')
    # plt.ylim(ylim)
    plt.xlabel('seconds')
    plt.title('z0 over time')
    plt.legend()
    plt.show()  
    
def plot_spatial(df_x, df_y, x, y):
    plt.figure()
    plt.scatter(df_x, df_y)
    plt.xlabel(f'{x} m')
    plt.ylabel(f'{y} m')
    # plt.gca().invert_xaxis()  # Reverse x-axis
    # plt.xlim(y_max, y_max)  # Set global limits
    # plt.ylim(x_min, x_max)  # Set global limits
    plt.title(f'{x} , {y} positional values of measurements', )
    plt.show()    
#%% PRINT FUNCTIONS
def print_value(name,type_data,df):
    """
    Prints the mean value for a specified column in df if there are more than one non-NaN entries.
    """
    # Drop NaN values before calculating the mean to avoid errors
    non_nan_values = df[name].dropna()
    if len(non_nan_values) > 0:
        print(f'Mean {type_data} ({name} fit) = {np.mean(non_nan_values):.3f}')
    else:
        print(f'Not enough data to calculate mean value {type_data} for {name} fit.')
        

        
def print_rmse():
    print('RMSE Rotor Plane')
    print_value('Power_Curved'  ,'rmse',rmse_df)
    print_value('Power_Linear'  ,'rmse',rmse_df)
    print_value('Log'           ,'rmse',rmse_df)
    print_value('Poly'          ,'rmse',rmse_df)
    print()
    
    print('RMSE Full Height (last fit)')
    print(f'Power (Curved) = {rmse_full_height_curved:.3}')
    print(f'Power (Linear) = {rmse_full_height_linear:.3}')
    print(f'Log = {rmse_full_height_log:.3}')
    print(f'Polynomial Fit {poly_degree} deg. = {rmse_full_height_poly:.3}')
    print()


    print(f'RMSE Binned Values (no fitting) = {rmse_binned_speeds_curved:.2f}')
    print()

        
def print_advection():    
    print('Last Advection speeds from each fit:')
    print(f'Power_Curved  = {bestAdvecPowerCurved:.3}')
    print(f'Power_Linear  = {bestAdvecPowerLinear:.3}')
    print(f'Log  = {bestAdvecLog:.3}')
    print(f'Poly  = {bestAdvecPoly:.3}')

def print_best_vals():
    """
    Prints the 'last' values for all the fits: Power_Curved, Power_Linear, Log, and Poly.
    """
    print()
    print_rmse()
    print()
    print_advection()
    print()
    
    
def print_all_means():
    """
    Prints the mean values for all the fits: Power_Curved, Power_Linear, Log, and Poly.
    """
    print()
    print('Last values for post processing and evaluation')
    print_rmse()
    print()
    print_advection()
    print()
    
def print_parameters():
    print('PARAMETERS')
    print(f'Bin Size = {bin_size} [m]')
    print(f'Timescale = {timescale} [s]')
    print(f'Distance to Estimation Plane = {dist_est_plane} [m]')
    print(f'Estimation Box Length = {est_box_len} [m]')
    print(f'Bin Size = {bin_size} [m]')
    print(f'Degree of Polynomial Fit = {poly_degree}')
    print(f'Sampling Period Length = {dt_samp} [s]')
    print(f'Sampling Frequency = {1/dt_samp:.2}')
    print()
    
def print_key_values():
    print('KEY VALUES: "Mean Last" ')
    print(f'Advection speed at {z_ref}m = {meanLastAdvec:.2}')
    print(f'RMSE= {meanLastRMSE:.3}')
    print(f'Alpha= {meanLastAlpha:.3}')
    

#%% Estimation_box_model function defintion and data preparation
def estimation_box_model():
    #1: filter out all measurements that did not occur between t-time_scale and t
    #create mask (select indices of time)
    time_mask = (df_active['Time']<=t) & (df_active['Time']>=t-timescale)
    #copy the dataframe to make a new dataframe only containing the valid indices
    df_time_masked = df_active[time_mask].copy()  # Use .copy() here to avoid warnings
        
 	#2: adjust y_g coordinate of "old" measurements using taylors frozen turbulence hypothesis
     #define the time between the measurement and the reference time "t"     
    dt_elapsed = t - df_time_masked['Time']  #( if we are at t=300 seconds, and the time of the measurement is t_meas=200, the elapsed time differenceis dt_elaps100 seconds )
    #create column with new positions y_new, shifted by the product of the advection speed and the elapsed time. 
    df_time_masked.loc[:, 'y_new'] = df_time_masked['y'] - dt_elapsed * u_adv #u_adv is first initialized as the mean wind speed, and later updated with each fit
        
 	#3: use y_(g,new)  to filter data close to esimation plane
    #create mask (select indices inside estimation box) 	
    distance_mask = (df_time_masked['y_new']>=dist_est_plane-est_box_len/2) & (df_time_masked['y_new']<=dist_est_plane+est_box_len/2)
    #copy the dataframe to make a new dataframe only containing the valid indices, so the df now only contains values of the estimation box
    df_distance_masked = df_time_masked[distance_mask].copy()  # Use .copy() here

 	#4: Group by vertical bin
    #use pd.cut to bin each height into a new column z_binned, based on z. 
    df_distance_masked.loc[:, 'z_binned'] = pd.cut(df_distance_masked['z'], bins, right=False) #bins are calculated in lines 196-199 since pd.cut wants either amount of bins or a list of bins, and we would rather give a bin width
    # Add a column for the midpoint of each bin (i.e., (left + right) / 2)
    df_distance_masked.loc[:, 'z_avg'] = df_distance_masked['z_binned'].apply(lambda x: (x.left + x.right) / 2)
    #Group by 'z_binned' to calculate the mean wind speed for each bin
    df_grouped = df_distance_masked.groupby('z_binned').agg(
    u=('v', 'mean'),
    z_avg=('z_avg', 'first'),  # We use 'first' because all rows in the group have the same z_mid value    
    count=('v', 'size')  # You can also count how many data points are in each bin
    ).reset_index()  # Reset index so the 'z_binned' becomes a normal column again
    
    # # Step 3: Merge the bin statistics (mean wind speed) back to the original df_distance_masked
    df_merged = pd.merge(df_distance_masked, df_grouped, on='z_binned', how='left')   
    # Remove the duplicate z_avg column and rename the remaining one
    df_merged.drop(columns=['z_avg_y'], inplace=True)  # Remove 'z_avg_y'
    df_merged.rename(columns={'z_avg_x': 'z_avg'}, inplace=True)  # Rename 'z_avg_x' to 'z_avg'
    
    return df_time_masked, df_distance_masked, df_grouped, df_merged

def check_before_fit(df):
    # Ensure z_avg_numeric is numeric before fitting
    df['z_avg_numeric'] = pd.to_numeric(df['z_avg'], errors='coerce')
    
    # Drop rows with NaN values in z_avg_numeric or u
    df = df.dropna(subset=['z_avg_numeric', 'u'])
    return df


#%% TRY FIT AND PLOT FUNCTIONS
def tryFitOld(fit):
    try:
        # Call the power_curved_fit function with valid data
        fitted_speeds, rmse, parameters = fit(
            z=df_grouped['z_avg_numeric'], 
            v=df_grouped['u'],             
        )        
        return fitted_speeds, rmse, parameters
    except Exception as e:
        print(f"Error fitting at t={t}: {e}")
        return np.nan, np.nan, [np.nan,np.nan]

def tryFit(fit):
    try:
        # Call the power_curved_fit function with valid data
        fitted_speeds, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, parameters = fit(
            z=averaged_df['z_avg_numeric'], 
            v=averaged_df['u'],             
        )        
        return fitted_speeds, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, parameters
    except Exception as e:
        print(f"Error fitting at t={t}: {e}")
        return np.nan, np.nan,np.nan, np.nan,  [np.nan,np.nan]
    
def tryFitPower(fit,alpha,v_ref):
    try:
        # Call the power_curved_fit function with valid data
        fitted_speeds, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, parameters = fit(
            z=averaged_df['z_avg_numeric'], 
            v=averaged_df['u'],
            alpha=alpha, v_ref = v_ref                  
        )        
        return fitted_speeds, rmse_rotor_plane, rmse_full_height, rmse_binned_speeds, parameters
    except Exception as e:
        print(f"Error fitting at t={t}: {e}")
        return np.nan, np.nan,np.nan, np.nan,  [np.nan,np.nan]
    
    

def tryPlotOld(fitted_speeds, fit_name, rmse, tryMarker):
    if tryMarker == True:
        try:
            # Plot the vertical wind profile with fitted speeds (power law)
            plot_vertical_fit(
                df_grouped['u'], df_grouped['z_avg_numeric'], 'binned speeds',
                fitted_speeds, df_grouped['z_avg_numeric'], f'{fit_name} fit',
                f'Binned speeds with {fit_name} fit. t={t}s', float(rmse), 
                bin_size, 'Bin size: [m]'
            )
        except Exception as e:
            print(f"Error plotting power law fit at t={t}: {e}")
            
def tryPlot(fitted_speeds, fit_name, rmse, tryMarker):
    heights = np.linspace(z_min,z_max,z_range[1])
    if tryMarker == True:
        try:
            # Plot the vertical wind profile with fitted speeds (power law)
            plot_vertical_fit(
                averaged_df['u'], averaged_df['z_avg_numeric'], 'binned speeds',
                fitted_speeds, heights, f'{fit_name} fit',
                f'{chosen_v}  {fit_name} fit. t={t:.0f}s', float(rmse), 
                bin_size, 'Bin size: [m]'
            )
        except Exception as e:
            print(f"Error plotting power law fit at t={t}: {e}")
# =============================================================================
#%% Parameters: BINS, TIMESCALES, DISTANCE, Estimation Box etc.
# # =============================================================================
# =============================================================================
# Define dataframe to be used
# =============================================================================

# df_active = df_original.copy()
df_active = df_v_y_extracted.copy()

# =============================================================================
# Define constants
# =============================================================================
D = 178
###########BIN SIZE#################
bin_size = 11 # Height bin size in meters
###########TIME SCALE###############
timescale = 60*10 # Time scales in seconds
###########Distance to Estimation Plane###############
dist_est_plane = 1*D #meters
###########Estimation Box Length###############
est_box_len = 2*D #m
#degree of polynomial fit
poly_degree = 2
#initialize mean advection speed u_adv and V_ref
u_adv_initial = 11.4
# u_adv_true_speeds = [7.98,10.53,13.06,15.54,15.54]
u_adv_true_speeds = [u_adv_initial]*8
u_adv_true = u_adv_true_speeds[0]
V_ref_true = u_adv_true 
u_adv = u_adv_initial
V_ref_initial = u_adv_initial
V_ref = V_ref_initial
###########Reference height (height of lidar position)###############
z_ref = 119 #m
rotor_lowest_height = z_ref-D/2 #we evaluate the fit starting from this height
rotor_highest_height = z_ref+D/2 #we evaluate the fit until this height
u_adv_true = u_adv_true_speeds[0]
# trueAlphas = [0.2,0.2,0.15,0.125,0.1,0.1,0.1]
trueAlphas = [0.2]*10
trueAlpha = trueAlphas[0]
estimationCounter = 0 #to change alpha or u_adv each 15 min
alpha_initial_guess = 0.2


#Decide which plots to run (T / F)
#Fits
plot_Power_Curved                 = False
plot_Power_Linear                 = False
plot_Log                          = False
plot_Poly                         = False
#Estimation Box
plot_vertical_m                   = False
plot_vertical_w_true_power        = True
plot_y_vs_z                       = False
plot_x_vs_y                       = False
plot_x_vs_z                       = False
plot_no_bins                      = False

# =============================================================================
# Construct time array with limits t_start and t_stop, at a period length of dt_samp
# =============================================================================
T_average = 30 # s
dt_samp = 5 # period length in seconds

t_start = df_v_y_extracted['Time'].min()  # seconds
t_stop = df_v_y_extracted['Time'].max()# seconds
# t_stop = t_stop+500
T_series = t_stop-t_start
t_fill_up = 100 #after 150 seconds the array is full and the algorith is as accurate as it can be
buffer = 0
t_cut_off =T_series+t_fill_up+buffer
t_last_average_time = 0
minutes_15 = 3600*(5500-3700)/8192
transient = 100/8192*3600
final_part = 3600*(8192-7300)/8192
alpha_changing_times = [final_part,minutes_15+final_part,2*minutes_15+final_part,3*minutes_15+final_part,4*minutes_15+final_part,transient+4*minutes_15+final_part,t_stop]
change_variable = 0
alpha_time = alpha_changing_times[change_variable]

advec_changing_times = [970,1870,2770,t_stop]

changing_times = advec_changing_times

    

# Use np.arange to create the time array with a step size of dt_samp
time_arr = np.arange(t_start, t_stop + T_average, T_average)  # Added t_stop + dt_samp to include t_stop
#parameters for bin function (pd.cut)
z_min, z_max = df_active['z'].min(), df_active['z'].max()
#uniform bins, pd.cut needs amount of bins. can either be sequence of scalars or a number of bins. This means i can customize the bin size to be non-uniform if i want to
bins = np.arange(np.floor(z_min), np.ceil(z_max) + bin_size, bin_size)




#

# #check the data
plt.figure()
plt.plot(df_active['Time'],df_active['v'],linewidth=0.05)
plt.show()

# plt.scatter(alpha_changing_times,trueAlphas)
#check the data
# plt.figure()
# plt.plot(df_v_y_extracted['Time'],df_v_y_extracted['HuLi_V_LOS_wgh_scaled'],linewidth=0.05)
# plt.show()


binned_rmse_list=[]
#%% MAIN LOOP
for t in time_arr:
    # print(f't={t}')    
    # print(f'binning and filtering at t={t} ')
    # =============================================================================
    # ESTIMATION BOX FILTERING FUNCTION
    # =============================================================================
    df_time_masked, df_distance_masked, df_grouped, df_merged = estimation_box_model()    #so now we have two dataframes:        #df_merged contains all the values as df, but only in the estimation box, and all the values are binned as well. This is mainly for validation        #df_grouped contains the important values: wind speed and heights (and a count), which is what we need for the vertical wind profile    
    # =============================================================================
    # PLOTS:    ESTIMATION BOX
    # =============================================================================
    plot_estimation_box_y_vs_z(plot_y_vs_z)  # WITH BINS (side view y vs z)
    plot_estimation_box_x_vs_y(plot_x_vs_y)  #estimation box x and y values (top view)    
    plot_estimation_box_x_vs_z(plot_x_vs_z)  #estimation box x and z values (lidar POV)      
    plot_estimation_box_no_bins(plot_no_bins) # WITHOUT BINS
   
    # plot_counts_barplot(df_grouped) #plot distribution of measurements at each binned height
    
    updatePointsCounter() #just to see amount of points in estimation box at each timestep
          
   
        
        
    df_grouped = check_before_fit(df_grouped) #prepare column for filtering (check for negative and NaNs)
    # Now check if there are valid rows left
    if not df_grouped['z_avg_numeric'].empty and not df_grouped['u'].empty:
        #Store the binned values
        # print(f'Storing binned values at t={t} ')
        # storage_df = pd.concat([storage_df, df_grouped['u'].reset_index(drop=True)],axis=1)        
        # Add the current timestamp as the column name for the new column
        storage_df = pd.concat([storage_df, df_grouped['u'].reset_index(drop=True).rename(f'{t}')], axis=1)
        # Debug: Print column names after adding the new column
        # print(f"Columns in storage_df after appending (t={t}): {storage_df.columns.tolist()}")


        # Proceed only if t is within the range
        if t > t_fill_up+t_start and t < t_cut_off:
            #perform fitting at a lower frequency than binning (T_average)
            if t-t_last_average_time >= T_average:
                # print('############## ESTIMATION ##############')
                # print(f"Averaging binned values every {T_average}s, with a window of {timescale}s and fitting at t={t}")
                # Calculate the mean speeds for each height from storage_df
                
                
                # Filter storage_df to keep only recent measurements within the last `timescale` seconds
                if not storage_df.empty:
                    # Debug: Print column names before filtering
                    # print(f"Data before filtering. Time range: {storage_df.columns.tolist()[0]} ... {storage_df.columns.tolist()[-1]} ")

                    # print(f"Columns in storage_df before filtering (t={t}): {storage_df.columns.tolist()[0]} ... {storage_df.columns.tolist()[-1]} ")
    
                    storage_df = storage_df.loc[:, [col for col in storage_df.columns if float(col) >= t - timescale]]
    
                    # Debug: Print column names after filtering
                    # print(f"Columns in storage_df after filtering (t={t}): {storage_df.columns.tolist()[0]} ... {storage_df.columns.tolist()[-1]} ")
                    # print(f"After filtering, fitting with data in time range: {storage_df.columns.tolist()[0]} ... {storage_df.columns.tolist()[-1]} ")

                    # print()
                
                try:
                    mean_speeds = storage_df.mean(axis=1)
                    # print(f' mean_speeds length {len(mean_speeds)}')
                    # print(f'len(df_grouped["z_avg_numeric"]) = {len(df_grouped["z_avg_numeric"])}') 
                    # Create a dataframe with heights and averaged speeds
                    averaged_df = pd.DataFrame({
                        "z_avg_numeric": df_grouped["z_avg_numeric"],  # Assuming heights are constant
                        "u": mean_speeds
                        })
                    
                except Exception as e:
                        print(f"Error creating averaged_df {e}")
                        # return np.nan, np.nan,np.nan, np.nan,  [np.nan,np.nan]
                    
                
               
                
                if t>=changing_times[change_variable]:
                    change_variable +=1
                    trueAlpha=trueAlphas[change_variable]
                    u_adv_true=u_adv_true_speeds[change_variable]
                    
                v_ref_binned, lowerHeightSpeed,lowerHeight,upperHeightSpeed,upperHeight = getSpeedAtZ_ref()
                # v_ref_binned = linInterp(lowerHeight, upperHeight, lowerHeightSpeed, upperHeightSpeed, z_ref)
                # print(f'v_ref_binned = {v_ref_binned:.2f} [u(z_ref)]' )               
                # =============================================================================
                # PLOTS:    VERTICAL WIND PROFILE
                # =============================================================================
                
                # print(len(averaged_df['u']))

                # ====================================================
                #             POLY FIT
                # =============================================================================
                poly_fitted_speeds,rmse_rotor_plane_poly, rmse_full_height_poly, rmse_binned_speeds_poly, poly_coeffs= tryFit(polynomial_fit) #DO THE FIT
                relative_rmse_full_height_poly = rmse_rel(rmse_full_height_poly,'poly')
                tryPlot(poly_fitted_speeds,'polynomial', rmse_rotor_plane_poly, plot_Poly) #PLOT THE FIT  
                # print(len(poly_fitted_speeds))
                print(f't = {t}')
                print(f'Updating advection speed with polynomial fit from {u_adv:.2f} m/s to {np.polyval(poly_coeffs, z_ref):.2f} m/s')
                print()
                u_adv = np.polyval(poly_coeffs, z_ref)
                V_ref = u_adv
                # =============================================================================
                #             POWER CURVED FIT 
                # =============================================================================            
                power_curved_fitted_speeds, rmse_rotor_plane_curved, rmse_full_height_curved, rmse_binned_speeds_curved, alpha_curved = tryFitPower(power_curved_fit,alpha_initial_guess,V_ref) #DO THE FIT
                relative_rmse_full_height_curved = rmse_rel(rmse_full_height_curved,'curved')
                tryPlot(power_curved_fitted_speeds,'power law curved', rmse_full_height_curved, plot_Power_Curved) #PLOT THE FIT                
                # =============================================================================
                #             POWER LINEAR FIT 
                # =============================================================================
                power_linear_fitted, rmse_rotor_plane_linear, rmse_full_height_linear, rmse_binned_speeds_linear, alpha_linear= tryFitPower(power_fit_linearized,alpha_initial_guess,V_ref) #DO THE FIT
                relative_rmse_full_height_linear = rmse_rel(rmse_full_height_linear,'linear')
                tryPlot(power_linear_fitted,'power law linearized', rmse_rotor_plane_linear, plot_Power_Linear) #PLOT THE FIT
                # =============================================================================
                #             LOG FIT
                # =============================================================================
                log_fitted_speeds,rmse_rotor_plane_log, rmse_full_height_log, rmse_binned_speeds_log, log_parameters= tryFit(logarithmic_fit) #DO THE FIT
                relative_rmse_full_height_log = rmse_rel(rmse_full_height_log,'log')
                tryPlot(log_fitted_speeds, 'log', rmse_rotor_plane_log, plot_Log) #PLOT THE FIT
                
                # =============================================================================
                # PLOTS:    VERTICAL WIND PROFILE
                # =============================================================================
                # plot_vertical_specific(averaged_df['u'], averaged_df['z_avg_numeric'], plot_vertical_m) # plot the vertical wind profile with storage_df (average values for each bin size over T_averaged seconds only)    
                # plot_vertical_true_power(averaged_df['u'], averaged_df['z_avg_numeric'], plot_vertical_w_true_power) # plot the vertical wind profile with storage_df (average values for each bin size over T_averaged seconds only)    
                plot_vertical_specific(averaged_df['u'], averaged_df['z_avg_numeric'], plot_vertical_m) # plot the vertical wind profile with storage_df (average values for each bin size over T_averaged seconds only)    
                plot_vertical_true_power(averaged_df['u'], averaged_df['z_avg_numeric'], plot_vertical_w_true_power) # plot the vertical wind profile with storage_df (average values for each bin size over T_averaged seconds only)    


                # =============================================================================
                # 7: Update mean advection speed. E.g., u ̂_adv=u ̂(z_ref)
                # =============================================================================
                advection_df, rmse_df, params_df = updateAllDfs() #update post processing dataframes (rmse's, parameters, mean advection speeds)
                # u_adv=calculateMeanAdvectionSpeeds() #store the mean from all models
                u_adv = np.mean(advection_df)
                print('Fitting complete')
                print(f'Updating u_adv with mean u at {z_ref} m at each fit to {u_adv:.2f} m/s')
                print(f"Mean advection wind speed over whole height range= {np.mean(averaged_df['u']):.2f}")
                print()
                
                V_ref = u_adv   #apply it to reference speed 
                
                # Update the last average time and alpha
                t_last_average_time = t
                
                binned_rmse_list.append(rmse_binned_speeds_curved)
                    
                # estimationCounter +=1
                # if estimationCounter <=3:
                    
    
                
                
                
                
        else:
            print(f"df not saturated t={t}") #if we are outside fillup or cutoff time
    
    
#%% POST PROCESSING
# storage_df['speeds'] = storage_df.mean(axis=1)
# # Keep only 'speeds' and 'heights' columns
# storage_df = storage_df[["speeds", "heights"]]
# heights = np.linspace(z_range[0]+bin_size/2,z_range[1]+bin_size/2,int((z_range[1] - z_range[0]) / bin_size) + 1)
# storage_df['heights'] = heights[0:-1]

# plot_vertical(storage_df['speeds'], storage_df['heights'], 'average vertical profile', 'average vertical profile')


#clean up dataframes with rmse, advection speeds and parameters
cleanDf()

#Calculate key parameters
#these are means and are not accurate as the last fitted values are probably the most accurate, since they are fits for the whole dataset, while the initial fits only contain 120 seconds
meanAlpha, bestAlphaCurved, bestAlphaLinear, meanLastAlpha   = calcMeanAlpha()
meanRMSE, meanLastRMSE, bestRMSEPowerCurved, bestRMSEPowerLinear, bestRMSELog, bestRMSEPoly    = calcMeanVal(rmse_df)
meanAdvec,meanLastAdvec, bestAdvecPowerCurved, bestAdvecPowerLinear, bestAdvecLog, bestAdvecPoly   = calcMeanVal(advection_df)
# meanAdvec   = u_adv


# =============================================================================
# INFORMATIONAL PRINTS
# =============================================================================
print('####################################################')
print(f'chosen wind speed: {chosen_v}')

print_all_means() #rmse, advection speeds
# print_parameters() # bin size, estimation box parameters,sampling size etc.
print_key_values()
# =============================================================================
# POST PROCESS PLOTS (HISTORICAL TIMESERIES)
# =============================================================================
# plot_z0_vs_time([0,.75])
# plot_parameters_vs_time([0.05,0.3])
# plot_amount_of_points_vs_time()
# plot_mean_value_vs_time(rmse_df,'RMSE',[0,0.4],legend=True,legendPos='upper right')
# plot_mean_value_vs_time(advection_df,'u_adv_mean',[u_adv_initial-2,u_adv_initial+2],u_adv_true, mean=True, legend=True)

times = np.linspace(df_active['Time'].min(),df_active['Time'].max(),len(params_df))
# alpha_changing_times_plot = [0,1183,1184,1974,1975,2765,2766,3700]
alpha_changing_times_plot = [times[0],times[-1]]

# true_alphas_plot = [0.2,0.2,0.15,0.15,0.125,0.125,0.1,0.1]
true_alphas_plot = [0.2]*(len(alpha_changing_times_plot))
params_df['average_alpha'] = params_df[['alpha_linear', 'alpha_curved']].mean(axis=1)
# plt.figure()
# # plt.plot(times,params_df['alpha_linear'],label='alpha_linear')
# plt.plot(times,params_df['alpha_curved'],label = 'alpha',linewidth=3)
# # plt.plot(times,params_df['average_alpha'], label = 'average fitted alpha')
# plt.plot(alpha_changing_times_plot,true_alphas_plot,label='Expected Alpha', linestyle = ':',linewidth=3)
# # plt.ylim(0.05,0.3)
# plt.xlabel('seconds')
# plt.ylabel('alpha')

# plt.legend()
# plt.title(f'Alpha Convergence. Timescale = {timescale} s')

# plt.show()

times = np.linspace(df_active['Time'].min(),df_active['Time'].max(),len(advection_df))
advec_changing_times_plot = [0,40,80,940,970,1000,1840,1870,1900,2740,2770,2800,3700,4000]
# true_advec_plot = [4.56,np.mean([7.98,4.56]),7.98,7.98,np.mean([7.98,10.53]),10.53,10.53,np.mean([10.53,13.06]),13.06,13.06,np.mean([15.54,13.06]),15.54,15.54,15.54]
true_advec_plot = [u_adv_initial]*len(advec_changing_times_plot)
advection_df['average_advec'] = advection_df[['Log', 'Poly']].mean(axis=1)

height_mask = (df_v_y_extracted['z'] > 110) & (df_v_y_extracted['z'] < 130 )
df_timelist_plot = df_v_y_extracted[height_mask].reset_index(drop=True)




print_value('alpha_curved','alpha',params_df)
print_value('average_advec','U_estimate',advection_df)
print_value('average_advec','U_estimate',advection_df)


#%%imports
# from result_section_plots_timescale600 import *
# from result_section_plots_timescale540 import *
# from result_section_plots_timescale480 import *
# from result_section_plots_timescale420 import *
# from result_section_plots_timescale360 import *
# from result_section_plots_timescale300 import *
# from result_section_plots_timescale240 import *
# from result_section_plots_timescale180 import *
# from result_section_plots_timescale180 import *


#%%alpha convergence
# plt.figure()

# timescales = [180, 240, 300, 360, 420, 600]  # Timescales from 180 to 600 in 60-second increments
# alphas = [alpha180, alpha240, alpha300, alpha360, alpha420, alpha600]  # Corresponding alpha values
# linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), '-']  # Ensure alpha600 gets "-"
# colors = ['blue', 'green', 'orange', 'purple', 'red', 'black']  # Distinct colors for each timescale

# for timescale, alpha, linestyle, color in zip(timescales, alphas, linestyles, colors):
#     plt.plot(times, alpha, label=f'{timescale // 60} min', linewidth=2, linestyle=linestyle, color=color)

# plt.plot(alpha_changing_times_plot, true_alphas_plot, label='True', linestyle=':', linewidth=3, color='gray')
# plt.ylim(0.075, 0.23)
# plt.xlim(100, 680)
# plt.xlabel('seconds')
# plt.ylabel('alpha')
# plt.legend(loc='lower left')
# plt.title('Alpha Convergence with Various Timescales')
# plt.show()


#%% mean advec

# timescales = [180, 240, 300, 360, 420, 600]  # Timescales from 180 to 600 in 60-second increments
# ulist = [u180, u240, u300, u360, u420, u600]  # Corresponding u values


# plt.figure()
# for timescale, u, linestyle, color in zip(timescales, ulist, linestyles, colors):
#     plt.plot(times, u, label=f'{timescale // 60} min', linewidth=2, linestyle=linestyle, color=color)  # Explicitly pass color

# plt.plot(advec_changing_times_plot, true_advec_plot, label='True', linestyle=':', linewidth=3, color='gray')  # Add color to "True" line
# plt.plot(df_timelist_plot['Time']-25, df_timelist_plot['v'], color='black',  linewidth = .5,label='U: 110m-130m',alpha=0.1)
# plt.plot(advec_changing_times_plot,true_advec_plot,label='"True Mean Advection Speed"',linestyle='--',linewidth=4,color='orange')
# # plt.ylim(df_active['v'].min(),df_active['v'].max())
# # plt.ylim(10.5, 12.25)
# plt.ylim(8, 14)
# plt.xlim(40, 680)
# plt.xlabel('seconds')
# plt.ylabel('Wind Speed [m/s]')
# # plt.legend(loc='lower left')
# plt.title('U Convergence with Various Timescales')
# plt.show()

#%%mean advec

# timescales = [60,420]  # Timescales from 180 to 600 in 60-second increments
# ulist = [u60,u420]  # Corresponding u values


# plt.figure()
# for timescale, u, linestyle, color in zip(timescales, ulist, linestyles, colors):
#     plt.plot(times, u, label=f'{timescale // 60} min', linewidth=3, linestyle=linestyle, color=color)  # Explicitly pass color

# plt.plot(advec_changing_times_plot, true_advec_plot, label='True', linestyle=':', linewidth=3, color='black')  # Add color to "True" line
# plt.plot(df_timelist_plot['Time']-25, df_timelist_plot['v'], color='grey',  linewidth = .5,label='U: 110m-130m',alpha=0.5)
# # plt.plot(advec_changing_times_plot,true_advec_plot,label='"True Mean Advection Speed"',linestyle='--',linewidth=3,color='orange')
# # plt.ylim(df_active['v'].min(),df_active['v'].max())
# # plt.ylim(10.5, 12.25)
# plt.ylim(6, 16)
# plt.xlim(40, 680)
# plt.xlabel('seconds')
# plt.ylabel('Wind Speed [m/s]')
# plt.legend(loc='lower left')
# plt.title('U Estimation on short and long timescale')
# plt.show()


#%%rmse

# timescales = [180, 240, 300, 360, 420, 600]  # Timescales from 180 to 600 in 60-second increments
# rmseBinned = [binned_rmse_list180, binned_rmse_list240, binned_rmse_list300, binned_rmse_list360, binned_rmse_list420, binned_rmse_list600]  # Corresponding u values


# plt.figure()
# for timescale, rmse, linestyle,color in zip(timescales, rmseBinned,linestyles,colors):
#     plt.plot(times, rmse, label=f'{timescale // 60} min', linewidth=2,linestyle=linestyle)

# # plt.plot(advec_changing_times_plot, true_advec_plot, label='True', linestyle=':', linewidth=3)
# # plt.ylim(10.5, 12.25)
# plt.xlim(40, 680)
# plt.xlabel('seconds')
# plt.ylabel(' rmse [m/s]')
# plt.legend(loc='lower left')
# plt.title('Filtered RMSE Convergence with Various Timescales')
# plt.show()


# plot_mean_value_vs_time(advection_df,'u_adv_mean',[u_adv_initial-2,u_adv_initial+2],u_adv_true, mean=True, legend=True)




#%%


# plt.figure()
# # plt.plot(times,advection_df['Power_Curved'],label='alpha_linear',linewidth=4)
# # plt.plot(times,advection_df['Power_Linear'],label = 'Power_Linear',linewidth=4)
# # plt.plot(times,advection_df['Log'], label = 'Log',linewidth=4)
# # plt.plot(xlist_for_plot, ylist_for_plot, color='black',  linewidth = .1,label='U: Beam 1')
# plt.plot(df_timelist_plot['Time']-25, df_timelist_plot['v'], color='black',  linewidth = .5,label='U: 110m-130m')
# plt.plot(times,advection_df['average_advec'], label = 'average_advec',linewidth=5)
# plt.plot(advec_changing_times_plot,true_advec_plot,label='"True Mean Advection Speed"',linestyle='--',linewidth=4,color='orange')
# # plt.plot([], [], ' ', label=f"Average = {np.mean(advection_df['average_advec']):.3f}")    
# # plt.plot([], [], ' ', label=f"True mean advection speed = {np.mean(true_advec_plot):.3f}")    
# plt.ylim(df_active['v'].min(),df_active['v'].max())
# plt.xlabel('Time [s]')
# plt.ylabel('Wind Speed [m/s]')
# plt.xlim(df_timelist_plot['Time'].min(),df_timelist_plot['Time'].max()-25)
# plt.legend()
# plt.show()

#%%



#%%
# Residual Analysis for Power Law
def residuals(fitted_speeds):
    truePowerLawSpeeds = truePowerLaw()[0]
    residuals = fitted_speeds - truePowerLawSpeeds
    return residuals
residuals_power_law_curved = residuals(power_curved_fitted_speeds)
residuals_power_law_linear = residuals(power_linear_fitted)
residuals_log = residuals(log_fitted_speeds)
residuals_poly= residuals(poly_fitted_speeds)


# relative_residuals = residuals_power_law/power_curved_fitted_speeds 



relative_rmse_full_height_curved = rmse_rel(rmse_full_height_curved,'curved')
relative_rmse_full_height_linear = rmse_rel(rmse_full_height_linear,'linear')
relative_rmse_full_height_log = rmse_rel(rmse_full_height_log,'log')
relative_rmse_full_height_poly = rmse_rel(rmse_full_height_poly,'poly')

# #%% Residual analysis
# def plot_residuals(fit_name, residuals, fitted_values):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(f'Residual Analysis for {fit_name} Fit')

#     # Normal Probability Plot
#     stats.probplot(residuals, dist="norm", plot=axs[0, 0])    
#     # Adjust marker size in the probability plot
#     for line in axs[0, 0].get_lines():
#         line.set_markersize(5)    
#     axs[0, 0].set_title("Normal Probability Plot")
#     # axs[0, 0].set_ylim(-.25, .25)  # Adjust y-axis limits here

#     # Residuals vs Fitted Values
#     axs[0, 1].scatter(fitted_values, residuals, color='red', s=8)
#     axs[0, 1].axhline(0, color='black', linestyle='--')
#     axs[0, 1].set_title("Residuals vs Fitted Values")
#     axs[0, 1].set_xlabel("Fitted Value")
#     axs[0, 1].set_ylabel("Residual")
#     # axs[0, 1].set_ylim(-1, 1)  # Set y-axis limits

#     # Histogram of Residuals
#     sns.histplot(residuals, kde=True,kde_kws={'bw_adjust': 0.1}, ax=axs[1, 0], color='gray', stat='percent')
#     axs[1, 0].set_title("Histogram of Residuals")
#     axs[1, 0].set_xlabel("Residual")
#     axs[1, 0].set_ylabel("Frequency")
#     # axs[1,0].set_xlim(-.025,.025)

#     # Residuals vs Observation Order
#     axs[1, 1].plot(residuals, marker='o', linestyle='-', color='blue', linewidth=0.1, markersize=3)
#     axs[1, 1].axhline(0, color='black', linestyle='--')
#     axs[1, 1].set_title("Residuals vs Observation Order")
#     axs[1, 1].set_xlabel("Observation Order")
#     axs[1, 1].set_ylabel("Residual")
#     # axs[1, 1].set_ylim(-1, 1)  # Set y-axis limits

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


# plot_residuals("Power Law", residuals_power_law, power_curved_fitted_speeds)
# plot_residuals("Power Law relative residuals", relative_residuals, power_curved_fitted_speeds)


#%% Model validation


def plot_residuals_new(fit_name, residuals, fitted_values):
    
    heights = np.linspace(z_min,z_max,z_range[1])
    truePowerLawSpeeds = power_law(heights, trueAlpha,u_adv_true)

    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Residual Analysis for {fit_name} Fit')

    # Normal Probability Plot
    stats.probplot(residuals, dist="norm", plot=axs[0, 0])    
    # Adjust marker size in the probability plot
    for line in axs[0, 0].get_lines():
        line.set_markersize(5)    
    axs[0, 0].set_title("Normal Probability Plot")
    # axs[0, 0].set_ylim(-1, 1)  # Adjust y-axis limits here
    
    # axs[0, 0].

    # Residuals vs Fitted Values
    axs[0, 1].scatter(fitted_values, residuals, color='red', s=8)
    axs[0, 1].axhline(0, color='black', linestyle='--')
    axs[0, 1].set_title("Residuals vs Fitted Values")
    axs[0, 1].set_xlabel("Fitted Value")
    axs[0, 1].set_ylabel("Residual")
    axs[0, 1].set_ylim(-1, 1)  # Set y-axis limits

    
    
    #True vs fitted vals
    slope,intercept,rval,pval,stderr = linregress(fitted_values,truePowerLawSpeeds)
    axs[1, 0].scatter(fitted_values,truePowerLawSpeeds,label='True vs fitted speeds',s=6,color='green',marker='o')
    axs[1, 0].plot(fitted_values,slope*fitted_values+intercept,label='regression line',c='black')
    axs[1, 0].set_title('True vs fitted speeds')
    axs[1, 0].set_xlabel("Fitted speeds")
    axs[1, 0].set_ylabel("True speeds")


    #residuals vs heights
    zeros = np.zeros(len(heights))

    
    axs[1, 1].scatter(heights, residuals,label='Residuals vs heights',s=6,color='green',marker='o')
    axs[1, 1].plot(heights,zeros,label='zero line',c='black')
    axs[1, 1].set_title('residuals vs heights')
    axs[1, 1].set_xlabel("Heights")
    axs[1, 1].set_ylabel("Residuals")
    axs[1, 1].set_ylim(-1,1)
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# plot_residuals_new("Power Law Curved", residuals_power_law_curved, power_curved_fitted_speeds)
# plot_residuals_new("residuals_power_law_linear", residuals_power_law_linear, power_linear_fitted)
# plot_residuals_new("residuals_log", residuals_log, log_fitted_speeds)
# plot_residuals_new("residuals_poly", residuals_poly, poly_fitted_speeds)

#%%
# # %matplotlib inline
# from statsmodels.graphics.gofplots import ProbPlot

# plt.style.use('seaborn') # pretty matplotlib plots

# plt.rc('font', size=14)
# plt.rc('figure', titlesize=18)
# plt.rc('axes', labelsize=15)
# plt.rc('axes', titlesize=18)

# QQ = ProbPlot(residuals_power_law_curved)
# plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

# plot_lm_2.set_figheight(8)
# plot_lm_2.set_figwidth(12)

# plot_lm_2.axes[0].set_title('Normal Q-Q')
# plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
# plot_lm_2.axes[0].set_ylabel('Standardized Residuals');





























