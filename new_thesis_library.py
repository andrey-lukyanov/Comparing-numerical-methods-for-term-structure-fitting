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

#Nelson-Siegel function
def g(m, tau):
    return (1 - np.exp(- m / tau)) / (m / tau)

def h(m, tau):
    return g(m, tau) - np.exp(- m / tau)


def ns(m, theta):
    tau = theta[0]
    beta0 = theta[1]
    beta1 = theta[2]
    beta2 = theta[3]

    return beta0 + beta1 * g(m, tau) + beta2 * h(m, tau)

#Reparameterized Nelson-Siegel function
def rep_ns(m, u):
    return u[1] + (u[2] - u[1]) * g(m, u[0]) + u[3] * h(m, u[0])

#Add prices discounted by nss curve to a dataframe on a date. Works specifically for build_ss_loss_function 
def discount(df, theta):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-ns(m = df['Дата фактической выплаты'], theta = theta) * df['Дата фактической выплаты'])
    
def discount_rep(df, u):
    df['Discounted'] = (df['Сумма купона, RUB'] + df['Погашение номинала, RUB']) * np.exp(-rep_ns(m = df.t, u = u) * df.t)
    
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


#loss_functions = [build_ss_loss_function(dates[date_number]) for date_number in range(len(dates))]

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

#constrained_loss_functions = [build_constrained_loss_function(dates[date_number]) for date_number in range(len(dates))]

def build_rep_loss_function_and_grad(date):
    
    market_prices = bonds_prices[date:date].T
    market_prices.columns = ['Market prices']
    market_prices.dropna(inplace=True)
    
    payments_on_date = bonds_payments[bonds_payments['Дата фактической выплаты'] >= date]
    payments_on_date = payments_on_date[payments_on_date['ISIN'].isin(market_prices.index)]

    calc_df = pd.concat([(payments_on_date['Дата фактической выплаты'] - date).apply(lambda x: x.days)/365, 
                          payments_on_date[['ISIN', 'Сумма купона, RUB', 'Погашение номинала, RUB']]], axis = 1)
    calc_df['CF'] = calc_df[['Сумма купона, RUB', 'Погашение номинала, RUB']].sum(axis = 1)
    calc_df.rename(columns = {'Дата фактической выплаты': 't'}, inplace = True)
    
    def loss_function(u):
        
        nonlocal calc_df, market_prices
        
        discount_rep(calc_df, u)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               market_prices], axis = 1)
        
        #Sum of squares
        J = (((np.array(result_df['Discounted']) - np.array(result_df['Market prices']))/1000)**2).sum()
       
        return J
    
    def gradient(u):
        
        nonlocal calc_df, market_prices
        
        discount_rep(calc_df, u)
      
        calc_prices = pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum())
                
        result_df = pd.concat([pd.DataFrame(calc_df.groupby('ISIN')['Discounted'].sum()), 
                               market_prices], axis = 1)


        result_df['price_diff'] = result_df['Market prices'] - result_df['Discounted']
        
        #partials calculation
        first_mult = calc_df.CF * calc_df.t * np.exp(-rep_ns(calc_df.t, u) * calc_df.t) / 500
                
        #d_tau calculation
        calc_df['tau_mult'] = first_mult * (u[1] + (u[2] - u[1] + u[3])/calc_df.t * (1 - np.exp(-calc_df.t / u[0]) * (1 + calc_df.t / u[0])) + u[3]/u[0]**2 * np.exp(-calc_df.t / u[0]))
        result_df['tau_mult'] = calc_df.groupby('ISIN')['tau_mult'].sum()
        d_tau = ((result_df['price_diff']) * result_df['tau_mult']).sum()        

        #du_0 calculation
        calc_df['du_0_mult'] = first_mult * (1 - g(calc_df.t, u[0]))
        result_df['du_0_mult'] = calc_df.groupby('ISIN')['du_0_mult'].sum()
        du_0 = ((result_df['price_diff']) * result_df['du_0_mult']).sum()
 
        #du_1 calculation
        calc_df['du_1_mult'] = first_mult * g(calc_df.t, u[0])
        result_df['du_1_mult'] = calc_df.groupby('ISIN')['du_1_mult'].sum()
        du_1 = ((result_df['price_diff']) * result_df['du_1_mult']).sum()
        
        #du_2 calculation
        calc_df['du_2_mult'] = first_mult * h(calc_df.t, u[0])
        result_df['du_2_mult'] = calc_df.groupby('ISIN')['du_2_mult'].sum()
        du_2 = (result_df['price_diff'] * result_df['du_2_mult']).sum()
           
        return np.array([d_tau, du_0, du_1, du_2])

    return loss_function, gradient
rep_loss_functions = [build_rep_loss_function_and_grad(dates[date_number]) for date_number in range(len(dates))]
    
#optimize on date by method with staring values
def optimize_on_day_with_starting_values(date_number, method, theta0):
    """
    a copy of theta0 should be supplied for L-BFGS-B and Powell, i.e. np.copy()
    """

    if method == 'L-BFGS-B':
        
        theta0[2] = theta0[2] + theta0[1]
        
        loss_func, grad = rep_loss_functions[date_number]
        
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
        
        loss_func, _ = rep_loss_functions[date_number]
        
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