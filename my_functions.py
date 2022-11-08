
# Import dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy


### statsmodels Functions

def smf_results_to_df(results):
    # Function takes the result of an statsmodel results table and transforms it into a dataframe
    # Appropriate usage: results = smf.ols().fit()
    p_value = results.pvalues
    coeff = results.params
    std_err = results.bse
    conf_lb = results.conf_int()[0]
    conf_ub = results.conf_int()[1]

    results_df = pd.DataFrame({"p_value":p_value,
                               "coeff":coeff,
                               "std_err":std_err,
                               "conf_lb":conf_lb,
                               "conf_ub":conf_ub
                                })

    #Reordering...
    results_df = results_df[["coeff","std_err","p_value","conf_lb","conf_ub"]]
    return results_df

def run_quantile_regressions(df, formula, q, varname):
    # Function runs multiple quantile regressions and returns a dataframe with the results for a selected vareter of interest
    # Arguments:
        # df: dataframe
        # formula: regression equation you wish to use, example "spend ~ treatment + gender + income"
        # q: The space between each quantile regression i.e. q=.1 would be every decile
        # varname: The variable in the regression equation whose estimated parameter you wish to return example: "income"   
    all_qr_results = pd.DataFrame()
    for q in np.arange(q, 1-q, q):
        qreg = smf.quantreg(formula, data=df).fit(q=q)
        temp = pd.DataFrame({'coefficient':varname,
                             'q': [q],
                             'coeff': [qreg.params[varname]], 
                             'std': [qreg.bse[varname]],
                             'p_value':[qreg.pvalues[varname]],
                             'ci_lower': [qreg.conf_int()[0][varname]],
                             'ci_upper': [qreg.conf_int()[1][varname]]})
        all_qr_results = pd.concat([all_qr_results, temp]).reset_index(drop=True)
    return(all_qr_results)


### Plotly Functions

def dual_yaxis_lineplot(x, y1, y2,plot_title, x_title, y1_title, y2_title):
    # Function will return a dual y axis lineplot 

    # initiate a dual y axes subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces: 1st and 2nd lines
    fig = (fig
            .add_trace(
                go.Scatter(x=x, y=y1, name= y1_title),
                secondary_y=False)
            .add_trace(
                go.Scatter(x=x, y=y2, name= y2_title),
                secondary_y=True)
            )

    # Add figure titles
    fig = (fig.update_layout(title_text= plot_title)
        # Set x-axis title
        .update_xaxes(title_text= x_title)
        # Set y-axes titles
        .update_yaxes(title_text= y1_title, secondary_y= False)
        .update_yaxes(title_text= y2_title, secondary_y= True)
            )
    return(fig)


### Federal Reserve Economic Data (FRED) Functions

def get_fred_data(fred_code, metric_name, geo):
    # Function to pull in a time series from FRED and convert to a tidy dataframe 
    # Arguments:
        # fred_code: code of the time series, can be found on the FRED website
        # metric_name: what you want the name of the metric to be in the resulting dataframe
        # geo: the geographical unit the series applies to e.g. 'alabama', 'USA','world' (note: this is already in the fred_code, speciyfing only makes it explicit in the df)
    data = fred.get_series(fred_code)
    data = (pd.DataFrame(data)
        .reset_index()
        .rename(columns = {'index':'date',0:metric_name})
        .assign(geo = geo,
                year = lambda x: x.date.dt.year,
                month = lambda x: x.date.dt.month)
        .astype({'geo':'category'})
            )    
    data = data[['date','year','month','geo',metric_name,]]
    return(data)

