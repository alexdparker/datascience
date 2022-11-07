






### statsmodels Functions

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

def plotly_renderer(rend = 'vscode')
        # Function sets the plotly image renderer for a session, useful if you want notebook results to display in github
            # for static images: 'svg' or 'png' (visible in github) 
            # for interactive images:'vscode' (not viisble in github)
    if plotly_renderer in ['svg','png']:
        print("You've chosen static images for plotly data visualizations, these should display in github")
    elif plotly_renderer == 'vscode':
        print("You've chosen vscode for the ploltly renderer, these images will not display in github but will be animated in a VScode notebook environment")
    else:
        print("please choose an appropriate plotly image renderer")
    return(rend)


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