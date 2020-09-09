def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'
    
    if isinstance(q, int):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], bins=quantiles, **kwargs)
    
    return bins.sort_index()

def get_or_request_training_predictions_from_model(df, model, data_subset=dr.enums.DATA_SUBSET.VALIDATION_AND_HOLDOUT):
    
    assert data_subset in ['all', 
                           'validationAndHoldout', 
                           'holdout', 
                           'allBacktests'], \
    'data_subset must be either all, validationAndHoldout, holdout, or allBacktests'
    
    project = dr.Project.get(model.project_id)
    
    if data_subset == dr.enums.DATA_SUBSET.HOLDOUT:
        project.unlock_holdout()

    try:
        predict_job = model.request_training_predictions(data_subset)
        training_predictions = predict_job.get_result_when_complete(max_wait=10000)

    except dr.errors.ClientError:
        prediction_id = [
            p.prediction_id
            for p in dr.TrainingPredictions.list(project.id)
            if p.model_id == model.id and p.data_subset == data_subset
        ][0]
        training_predictions = dr.TrainingPredictions.get(project.id, prediction_id)

    preds = training_predictions.get_all_as_dataframe(serializer='csv')

    df = df.merge(
            preds,
            how='left',
            left_index=True,
            right_on=['row_id'],
            validate='one_to_one',
    )
    df = df.loc[~np.isnan(df['row_id']), :].reset_index(drop=True)
    
    return df

def plot_lift_chart(df, model, target, bins, data_subset=dr.enums.DATA_SUBSET.HOLDOUT, weights=False):
    
    #Download Training Predictions
    preds = get_or_request_training_predictions_from_model(df, model, data_subset=data_subset)
    preds = preds.loc[~np.isnan(preds['prediction']),['row_id','class_1.0',target,'weight']].reset_index(drop=True)
    #preds.sort_values(by=['class_1.0'], ascending=True, inplace=True)
    
    #Create Quantiles
    if weights:
        preds['bins'] = weighted_qcut(preds['class_1.0'], preds['weight'], q=bins)
        mean = lambda x: np.average(x, weights=preds.loc[x.index, 'weight'])
    else:
        preds['bins'] = pd.qcut(preds['class_1.0'], q=bins)
        mean = lambda x: np.average(x)
        
    f = {'class_1.0':mean, target:mean}
    lc = preds.groupby(['bins']).agg(f).reset_index()
    lc['bins'] = range(1, bins+1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc['bins'], y=lc['class_1.0'],
                mode='lines+markers',
                name='Predictions',
                marker=dict(size=5,
                            color='blue',
                            symbol='cross-open',
                            line=dict(
                                    color='blue',
                                    width=2
                                    )
                           ),
                line=dict(
                    color='blue',
                    width=2
                        )
                    )
          )
    
    fig.add_trace(go.Scatter(x=lc['bins'], y=lc[target],
                        mode='lines+markers',
                        name='Actuals',
                        marker=dict(size=5,
                                    color='#ff7f0e',
                                    symbol='circle-open',
                                    line=dict(
                                        color='#ff7f0e',
                                        width=1
                                            )
                                   ),
                        line=dict(
                            color='#ff7f0e',
                            width=2
                                )
                            )
                  )
    
    title = 'Weighted ' if weights else ''
    fig.update_layout(title={
                            'text': f'<b>Lift Chart</b>',
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                              },
                      legend_title=f'Data: {data_subset}'
                     )
    fig.update_yaxes(title=f'{title}Average Target')
    fig.update_xaxes(title=f'{title}Bins')
    
    fig.show()