import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

path = '/Users/andrey_lukyanov/Google_Drive/Studies/Year_6/Thesis/Comparing-numerical-methods-for-term-structure-fitting/'
#path = 'C:/Users/1/Desktop/Comparing-numerical-methods-for-term-structure-fitting/'
#path = 'C:/Users/aaluk/Documents/GitHub/Comparing-numerical-methods-for-term-structure-fitting/'


#bonds_payments import
bonds_payments = pd.read_csv(path + 'Data/New_data/bonds_payments.csv', index_col = 0)
bonds_payments['Дата фактической выплаты'] = pd.to_datetime(bonds_payments['Дата фактической выплаты'], format='%Y-%m-%d')

#bonds_prices import
bonds_prices = pd.read_csv(path + 'Data/New_data/bonds_prices.csv', index_col='TRADEDATE', parse_dates=True)

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

def ns(m, theta):
    tau = theta[0]
    beta0 = theta[1]
    beta1 = theta[2]
    beta2 = theta[3]

    return beta0 + beta1 * (1 - np.exp(- m / tau)) / (m / tau) + beta2 * ((1 - np.exp(- m / tau)) / (m / tau) - np.exp(- m / tau))

def reparameterized_ns(m, u):
    return u[1] + (u[2] - u[1]) * (1 - np.exp(- m / u[0])) / (m / u[0]) + u[3] * ((1 - np.exp(- m / u[0])) / (m / u[0]) - np.exp(- m / u[0]))

#Add discounted by nss curve prices to a dataframe on a data. Works specifically for build_ss_loss_function 
def discount(df, theta):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-ns(m = df['Дата фактической выплаты'], theta = theta) * df['Дата фактической выплаты'])
    
def discount_rep(df, u):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-reparameterized_ns(m = df['Дата фактической выплаты'], u = u) * df['Дата фактической выплаты'])
    
#loss function
def build_ss_loss_function(date):

    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)    

    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
    def ss_loss_function(theta):
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount(calc_df, theta)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
        
        return J
    
    return ss_loss_function


loss_functions = [build_ss_loss_function(dates[date_number]) for date_number in range(len(dates))]

def tau_constraint(tau):
    if tau >= 30:
        return 1000 * (tau - 30)
    elif tau <= 0.01:
        return 1000 * (tau - 0.01)
    else:
        return 0
    
def beta_0_constraint(beta_0):
    if beta_0 <= 0:
        return beta_0
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
                               market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
        
        constraints = tau_constraint(theta[0])**2 + beta_0_constraint(theta[1])**2 + linear_constraint(theta[1], theta[2])**2
        
        return J + 1000 * constraints
    
    return loss_function

constrained_loss_functions = [build_constrained_loss_function(dates[date_number]) for date_number in range(len(dates))]

def build_reparametarized_loss_function(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)
    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
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
    """
    a copy of theta0 should be supplied for L-BFGS-B and Powell, i.e. np.copy()
    """

    if method == 'L-BFGS-B':
        
        theta0[2] = theta0[2] + theta0[1]
        
        loss_func = reparameterized_loss_functions[date_number]
        
        bounds = Bounds([0.01, 0, 0, -np.inf], 
                        [30, np.inf, np.inf, np.inf])
        
        start = dt.datetime.now()
    
        res = minimize(loss_func, theta0, method=method,
                       options={'disp': False}, bounds=bounds)
        
        execution_time = (dt.datetime.now() - start).total_seconds()
        
        theta = res.x
        theta[2] = theta[2] - theta[1]
        
        return theta, execution_time
    
    elif method == 'Powell':
        
        theta0[2] = theta0[2] + theta0[1]
        
        loss_func = reparameterized_loss_functions[date_number]
        
        bounds = ((0.01, 30), (0, 1), 
                  (0, 1), (-1, 1))
        
        start = dt.datetime.now()
    
        res = minimize(loss_func, theta0, method=method,
                       options={'disp': False}, bounds=bounds)
        print(res)

        execution_time = (dt.datetime.now() - start).total_seconds()
        
        theta = res.x
        theta[2] = theta[2] - theta[1]
        
        return theta, execution_time        
        
    elif method == 'trust-constr':      
        
        loss_func = loss_functions[date_number]
        
        bounds = Bounds([0.01, 0, -1, -1], 
                        [30, 1, 1, 1])
        linear_constraint = LinearConstraint([[0, 1, 1, 0]], [0], [np.inf])
        
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
    
def parallel_l_bfgs_b(starting_values):

    thetas = np.zeros([len(dates), 4])
    time = np.zeros(len(dates))

    for i in range(len(dates)):
        
        thetas[i], time[i] = optimize_on_day_with_starting_values(date_number = i, method = 'L-BFGS-B', theta0 = np.copy(starting_values.iloc[i][1:]))
        
    thetas = pd.DataFrame(thetas, index = dates, columns = ['tau', 'beta0', 'beta1', 'beta2'])
    thetas.to_csv(path + 'Thetas/l_bfgs_b_rand_' + str(int(starting_values.Value[0])) + '.csv')

    time = pd.DataFrame(time, index = dates, columns = ['Seconds'])
    time.to_csv(path + 'Time/l_bfgs_b_rand_' + str(int(starting_values.Value[0])) + '.csv')
    
                   
def parallel_powell(starting_values):

    thetas = np.zeros([len(dates), 4])
    time = np.zeros(len(dates))

    for i in range(len(dates)):
        
        thetas[i], time[i] = optimize_on_day_with_starting_values(date_number = i, method = 'Powell', theta0 = np.copy(starting_values.iloc[i][1:]))
        
    thetas = pd.DataFrame(thetas, index = dates, columns = ['tau', 'beta0', 'beta1', 'beta2'])
    thetas.to_csv(path + 'Thetas/powell_rand_' + str(int(starting_values.Value[0])) + '.csv')

    time = pd.DataFrame(time, index = dates, columns = ['Seconds'])
    time.to_csv(path + 'Time/powell_rand_' + str(int(starting_values.Value[0])) + '.csv')
                   
def parallel_nelder_mead(starting_values):

    thetas = np.zeros([len(dates), 4])
    time = np.zeros(len(dates))

    for i in range(len(dates)):
        
        thetas[i], time[i] = optimize_on_day_with_starting_values(date_number = i, method = 'Nelder-Mead', theta0 = np.copy(starting_values.iloc[i][1:]))
        
    thetas = pd.DataFrame(thetas, index = dates, columns = ['tau', 'beta0', 'beta1', 'beta2'])
    thetas.to_csv(path + 'Thetas/nelder_mead_rand_' + str(int(starting_values.Value[0])) + '.csv')

    time = pd.DataFrame(time, index = dates, columns = ['Seconds'])
    time.to_csv(path + 'Time/nelder_mead_rand_' + str(int(starting_values.Value[0])) + '.csv')
                   
def parallel_trust_constr(starting_values):

    thetas = np.zeros([len(dates), 4])
    time = np.zeros(len(dates))

    for i in range(len(dates)):        
        try:
            thetas[i], time[i] = optimize_on_day_with_starting_values(date_number = i, method = 'trust-constr', theta0 = np.copy(starting_values.iloc[i][1:]))           
        except ValueError:
            print('ValueError on starting value', int(starting_values.iloc[i][0]), 'on day', i)
            thetas[i] = np.nan
            time[i] = np.nan
        
    thetas = pd.DataFrame(thetas, index = dates, columns = ['tau', 'beta0', 'beta1', 'beta2'])
    thetas.to_csv(path + 'Thetas/trust_constr_rand_' + str(int(starting_values.Value[0])) + '.csv')

    time = pd.DataFrame(time, index = dates, columns = ['Seconds'])
    time.to_csv(path + 'Time/trust_constr_rand_' + str(int(starting_values.Value[0])) + '.csv')
    
def build_rmse_function(date):

    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)    

    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]
    
    def rmse_function(theta):
        
        nonlocal payments_on_date, market_prices, date
    
        calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
        
        discount(calc_df, theta)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               market_prices], axis = 1)
        
        #Sum of squares
        J = ((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))**2).mean()
        
        return J
    
    return rmse_function