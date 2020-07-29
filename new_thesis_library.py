import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

#/Users/andrey_lukyanov/Google_Drive/Studies/Year_4/Курсач/Coding/
#C:/Users/1/Desktop/
#C:/Users/aaluk/Documents/GitHub/

#bonds_payments import
bonds_payments = pd.read_csv('C:/Users/aaluk/Documents/GitHub/Comparing-numerical-methods-for-term-structure-fitting/Data/New_data/bonds_payments.csv', index_col = 0)
bonds_payments['Дата фактической выплаты'] = pd.to_datetime(bonds_payments['Дата фактической выплаты'], format='%Y-%m-%d')

#bonds_prices import
bonds_prices = pd.read_csv('C:/Users/aaluk/Documents/GitHub/Comparing-numerical-methods-for-term-structure-fitting/Data/New_data/bonds_prices.csv', index_col='TRADEDATE', parse_dates=True)

#dates and trade_codes
dates = bonds_prices.index
bond_isins = bonds_payments['ISIN'].unique()

#nss function
def nss(m, theta):
    tau1 = theta[0]
    tau2 = theta[1]
    beta0 = theta[2]
    beta1 = theta[3]
    beta2 = theta[4]
    beta3 = theta[5]
    return beta0 + beta1 * (1 - np.exp(- m / tau1)) / (m / tau1) + beta2 * ((1 - np.exp(- m / tau1)) / (m / tau1) - np.exp(- m / tau1)) + beta3 * ((1 - np.exp(- m / tau2)) / (m / tau2) - np.exp(- m / tau2))

def reparameterized_nss(m, u):
    return u[2] + (u[3] - u[2]) * (1 - np.exp(- m / u[0])) / (m / u[0]) + u[4] * ((1 - np.exp(- m / u[0])) / (m / u[0]) - np.exp(- m / u[0])) + u[5] * ((1 - np.exp(- m / u[1])) / (m / u[1]) - np.exp(- m / u[1]))    

#Add discounted by nss curve prices to a dataframe on a data. Works specifically for build_ss_loss_function 
def discount(df, theta):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-nss(m = df['Дата фактической выплаты'], theta = theta) * df['Дата фактической выплаты'])
    
def discount_rep(df, u):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-reparameterized_nss(m = df['Дата фактической выплаты'], u = u) * df['Дата фактической выплаты'])
    
#loss function
def build_ss_loss_function(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)
    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
    def ss_loss_function(theta):
        
        C = np.ones(8) * 1000
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount(calc_df, theta)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               corrected_market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
        
        return J
    
    return ss_loss_function


loss_functions = [build_ss_loss_function(dates[date_number]) for date_number in range(len(dates))]

def tau_constraint(tau):
    if tau >= 30:
        return tau - 30
    elif tau <= 0.05:
        return tau - 0.05
    else:
        return 0
    
def beta_0_constraint(beta_0):
    if beta_0 <= 0:
        return beta_0**2
    else:
        return 0
       
def linear_constraint(beta_0, beta_1):
    if beta_0 + beta_1 <= 0:
        return beta_0 + beta_1
    else:
        return 0

def build_constrained_loss_function(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)

    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
    def loss_function(theta):
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount(calc_df, theta)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               corrected_market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
        
        constraints = tau_constraint(theta[0])**2 + tau_constraint(theta[1])**2 + beta_0_constraint(theta[2])**2 + linear_constraint(theta[2], theta[3])**2
        
        return J + 1000 * constraints
    
    return loss_function

constrained_loss_functions = [build_constrained_loss_function(dates[date_number]) for date_number in range(len(dates))]

def build_reparametarized_loss_function(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)

    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
#    principals = pd.DataFrame(payments_on_date.groupby('ISIN')['Погашение номинала, RUB'].sum())
#    
#    #correction of amortizable bonds prices
#    corrected_market_prices = np.multiply(market_prices, principals) / 100
    
    def loss_function(u):
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount_rep(calc_df, u)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
       
        return J
    
    return loss_function

reparameterized_loss_functions = [build_reparametarized_loss_function(dates[date_number]) for date_number in range(len(dates))]
    
#optimize on date by method with staring values
def optimize_on_day_with_starting_values(date_number, method, theta0):

    if method == 'L-BFGS-B':
        
        theta0[3] = theta0[3] + theta0[2]
        
        loss_func = reparameterized_loss_functions[date_number]
        
        bounds = Bounds([0, 0, 0, 0, -np.inf, -np.inf], 
                        [30, 30, np.inf, np.inf, np.inf, np.inf])
        
        start = dt.datetime.now()
    
        res = minimize(loss_func, theta0, method=method,
                       options={'disp': False}, bounds=bounds)
        
        execution_time = (dt.datetime.now() - start).total_seconds()
        
        theta = res.x
        theta[3] = theta[3] - theta[2]
        
        return theta, execution_time
    
    elif method == 'Powell':
        
        theta0[3] = theta0[3] + theta0[2]
        
        loss_func = reparameterized_loss_functions[date_number]
        
        bounds = ((0, 30), (0, 30), (0, np.inf), 
                  (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        
        start = dt.datetime.now()
    
        res = minimize(loss_func, theta0, method=method,
                       options={'disp': False}, bounds=bounds)
        
        execution_time = (dt.datetime.now() - start).total_seconds()
        
        theta = res.x
        theta[3] = theta[3] - theta[2]
        
        return theta, execution_time        
    
    
    bounds = ((0, 30), (0, 30), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
        
    elif method == 'trust-constr':            
        loss_func = loss_functions[date_number]
        
        bounds = Bounds([0, 0, 0, -np.inf, -np.inf, -np.inf], 
                        [30, 30, np.inf, np.inf, np.inf, np.inf])
        linear_constraint = LinearConstraint([[0, 0, 1, 1, 0, 0]], [0], [np.inf])
        
        res = minimize(loss_func, theta0, method=method, 
                       constraints=[linear_constraint],
                       options={'disp': False}, bounds=bounds)
        return res.x, res.execution_time
           
    elif method == 'Nelder-Mead':
        loss_func = constrained_loss_functions[date_number]
        
        start = dt.datetime.now()
        
        res = minimize(loss_func, theta0, method=method,
                       options={'disp': False})
                
        execution_time = (dt.datetime.now() - start).total_seconds()
        
        return res.x, execution_time

    
def optimize_ss_bfgs(starting_values):

    thetas = np.zeros([len(dates), 6])

    theta0 = starting_values[1:]

    for i in range(len(dates)):
        
        thetas[i] = optimize_on_day_with_starting_values(date_number = i, method = 'BFGS', theta0 = theta0)
        
    thetas = pd.DataFrame(thetas, columns=['tau1', 'tau2', 'beta0', 'beta1', 'beta2', 'beta3'], index=dates)
    
    thetas.to_csv('C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/Thetas/bfgs_rand_' + str(int(starting_values[0])) + '.csv')