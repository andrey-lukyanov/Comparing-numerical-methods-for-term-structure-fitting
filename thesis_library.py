import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from scipy.optimize import least_squares
from pyswarm import pso


#/Users/andrey_lukyanov/Google_Drive/Studies/Year_4/Курсач/Coding/
#C:/Users/1/Desktop/

#bonds_payments import
bonds_payments = pd.read_csv('/Users/andrey_lukyanov/Google_Drive/Studies/Year_4/Курсач/Coding/Comparing-numerical-methods-for-term-structure-fitting/Data/bonds_payments.csv')
bonds_payments['Дата фактической выплаты'] = pd.to_datetime(bonds_payments['Дата фактической выплаты'])

#bonds_prices import
bonds_prices = pd.read_csv('/Users/andrey_lukyanov/Google_Drive/Studies/Year_4/Курсач/Coding/Comparing-numerical-methods-for-term-structure-fitting/Data/bonds_prices.csv', index_col='TRADEDATE', parse_dates=True)

#dates and trade_codes
dates = bonds_prices.index
bond_trade_codes = bonds_payments['Торговый код'].unique()

#nss function
def nss(m, theta):
    tau1 = theta[0]
    tau2 = theta[1]
    beta0 = theta[2]
    beta1 = theta[3]
    beta2 = theta[4]
    beta3 = theta[5]
    return beta0 + beta1 * (1 - np.exp(- m / tau1)) / (m / tau1) + beta2 * ((1 - np.exp(- m / tau1)) / (m / tau1) - np.exp(- m / tau1)) + beta3 * ((1 - np.exp(- m / tau2)) / (m / tau2) - np.exp(- m / tau2))

#add discounted prices to dataframe by nss curve
def discount(df, theta):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-nss(m = df['Дата фактической выплаты'], theta = theta)/100 * df['Дата фактической выплаты'])
    
#loss function
def build_ss_loss_function(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)

    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['Торговый код'].isin(market_prices.index)]
    
    principals = pd.DataFrame(payments_on_date.groupby('Торговый код')['Погашение номинала, RUB'].sum())
    
    #correction of amortizable bonds prices
    corrected_market_prices = np.multiply(market_prices, principals) / 100
    
    def ss_loss_function(theta):
        
        C = np.ones(8) * 1000
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['Торговый код', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount(calc_df, theta)
      
        calc_prices = pd.DataFrame(calc_df.groupby('Торговый код')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('Торговый код')['Discounted'].sum()), 
                               corrected_market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
        
        #constraints
        if (theta[0] > 0) & (theta[0] <= 30):
            c1 = 0
        else:
            c1 = C[1] * theta[0]**2

        if (theta[1] > 0) & (theta[1] <= 30):
            c2 = 0
        else:
            c2 = C[2] * theta[1]**2
            
        if theta[2] + theta[3] >= 0:
            c3 = 0
        else:
            c3 = C[3] * (theta[2] + theta[3])**2
            
        if (theta[2] >= 0) & (theta[2] <= 100):
            c4 = 0
        else:
            c4 = C[4] * theta[2]**2
        
        if (theta[3] >= -100) & (theta[3] <= 100):
            c5 = 0
        else:
            c5 = C[5] * theta[3]**2 

        if (theta[4] >= -100) & (theta[4] <= 100):
            c6 = 0
        else:
            c6 = C[6] * theta[4]**2 
            
        if (theta[5] >= -100) & (theta[5] <= 100):
            c7 = 0
        else:
            c7 = C[7] * theta[5]**2 
                            
        return J + c1 + c2 + c3 + c4 + c5 + c6 + c7
    
    return ss_loss_function


loss_functions = [build_ss_loss_function(date = dates[date_number]) for date_number in range(len(dates))]

#optimize on date by method with staring values
def optimize_on_day_with_starting_values(date_number, method, theta0):
    
    loss_func = loss_functions[date_number]
    
    if method != 'Gauss-Newton':
        res = minimize(loss_func, theta0, method=method, options={'xtol': 1e-8, 'disp': False, 'maxiter': 100000})
        
    else:
        res = least_squares(loss_func, theta0, method=method, options={'xtol': 1e-8, 'disp': False, 'maxiter': 100000})

    return res.x
    
def optimize_ss_bfgs(starting_values):

    thetas = np.zeros([len(dates), 6])

    theta0 = starting_values[1:]

    for i in range(len(dates)):
        
        thetas[i] = optimize_on_day_with_starting_values(date_number = i, method = 'BFGS', theta0 = theta0)
        
    thetas = pd.DataFrame(thetas, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates)
    
    thetas.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/bfgs_rand_' + str(int(starting_values[0])) + '.csv')
    
def optimize_ss_gauss_newton(starting_values):

    thetas = np.zeros([len(dates), 6])

    theta0 = starting_values[1:]

    for i in range(len(dates)):
        
        thetas[i] = optimize_on_day_with_starting_values(date_number = i, method = 'Gauss-Newton', theta0 = theta0)
        
    thetas = pd.DataFrame(thetas, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates)
    
    thetas.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/gauss_newton_rand_' + str(int(starting_values[0])) + '.csv')
    
def optimize_ss_nelder_mead(starting_values):

    thetas = np.zeros([len(dates), 6])

    theta0 = starting_values[1:]

    for i in range(len(dates)):
        
        thetas[i] = optimize_on_day_with_starting_values(date_number = i, method = 'nelder-mead', theta0 = theta0)
        
    thetas = pd.DataFrame(thetas, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates)
    
    thetas.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/nelder_mead_rand_' + str(int(starting_values[0])) + '.csv')

def optimize_ss_powell(starting_values):

    thetas = np.zeros([len(dates), 6])

    theta0 = starting_values[1:]

    for i in range(len(dates)):
        
        thetas[i] = optimize_on_day_with_starting_values(date_number = i, method = 'powell', theta0 = theta0)
        
    thetas = pd.DataFrame(thetas, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates)
    
    thetas.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/powell_rand_' + str(int(starting_values[0])) + '.csv')
    
def pso_multithread(interval):
    
    pso_batch = np.zeros([interval[1] - interval[0], 6])

    for i in range(interval[0], interval[1]):
        
        loss_func = loss_functions[i]
        
        lb = [0, 0, 0, -100, -100, -100]
        ub = [30, 30, 100, 100, 100, 100]
    
        res = pso(func=loss_func, lb = lb, ub = ub, maxiter=1000)
        
        pso_batch[i - interval[0]] = res[0]
        
    pso_batch = pd.DataFrame(pso_batch, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates[interval[0]:interval[1]])
    
    pso_batch.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/pso_batch_' + str(interval[1]) + '.csv')  
    
