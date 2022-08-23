#Import packages
import pandas as pd
import numpy as np
import datarobot as dr
import sys
import os
import random as rd
import re
import requests
#from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors

class partial_dependence_plots():
    """
    A class to calculate 1 or 2-way PDP plots
    
    Attributes:
    -----------
    df: DataFrame
    model: DataRobot Model Object
    project: DataRobot Project Object
    data_subset: str
        Partition to run calculations (CV, Validation, Holdout)
    sample: int
        Total number of rows to sample
    k: int
        Total number of values to sample per row
    
    
    Methods:
    --------
    
    
    """

    
    def __init__(self, 
                 df, 
                 model,
                 weight_col=None,
                 #data_subset=dr.enums.DATA_SUBSET.HOLDOUT, 
                 sample=1000, 
                 k=10):
        
        self.df = df
        self.model = model
        self.project = dr.Project.get(self.model.project_id)
        self.positive_class = self.project.positive_class
        #self.data_subset = data_subset
        self.target = self.project.target
        self.weights = weight_col
        self.sample = sample
        self.target_type = self.project.target_type
        self.k = k
        self.colors = [
                        '#1f77b4',  # muted blue
                        '#2ca02c',  # cooked asparagus green
                        '#d62728',  # brick red
                        '#ff7f0e',  # safety orange
                        '#9467bd',  # muted purple
                        '#8c564b',  # chestnut brown
                        '#e377c2',  # raspberry yogurt pink
                        '#7f7f7f',  # middle gray
                        '#bcbd22',  # curry yellow-green
                        '#17becf',  # blue-teal
                        '#ff7f0e'   # safety orange
                    ]
        self.opacities =['rgba(31, 119, 180, 0.1)', 'rgba(44, 160, 44, 0.1)',
                       'rgba(214, 39, 40, 0.1)', 'rgba(255, 127, 14, 0.1)', 
                       'rgba(148, 103, 189, 0.1)', 'rgba(140, 86, 75, 0.1)',
                       'rgba(227, 119, 194, 0.1)', 'rgba(127, 127, 127, 0.1)',
                       'rgba(188, 189, 34, 0.1)', 'rgba(23, 190, 207, 0.1)']
        self.features = []
        self.scores = pd.DataFrame()
        
    def _get_or_request_training_predictions_from_model(self, df, model, data_subset):
        """
        Get or request backtest and holdout scores from top models across multiple DataRobot projects

        df: DataFrame
        model:
            DataRobot model object
        data_subset: str (optional)
            Can be set to either allBacktests or holdout

        Returns:
        --------
        pandas df
        """
        assert data_subset in ['all', 
                               'validationAndHoldout', 
                               'holdout', 
                               'allBacktests'], \
        'data_subset must be either all, validationAndHoldout, holdout, or allBacktests'
        
        if data_subset == dr.enums.DATA_SUBSET.HOLDOUT:
            self.project.unlock_holdout()

        try:
            predict_job = model.request_training_predictions(data_subset)
            training_predictions = predict_job.get_result_when_complete(max_wait=10000)

        except dr.errors.ClientError:
            prediction_id = [
                p.prediction_id
                for p in dr.TrainingPredictions.list(self.project.id)
                if p.model_id == model.id and p.data_subset == data_subset
            ][0]
            training_predictions = dr.TrainingPredictions.get(self.project.id, prediction_id)

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
    
    def _create_scoring_data(self, df, model, feature_1, feature_2=None):
        """
        Create a synthetic scoring dataset
        
        Returns:
        --------
        pandas df
        
        """
        scoring_data = pd.DataFrame()
        sampled_rows = df.sample(n=min(self.df.shape[0],self.sample), replace=False)
        
        unique_values = df[feature_1].unique() # find unique values
        size = min(len(unique_values), self.k)
        f1_levels = df[feature_1].value_counts().iloc[0:size].index.values # grab the top k most common values
        
        if feature_2:
            unique_values = df[feature_2].unique() # find unique values
            size = min(len(unique_values), self.k)
            f2_levels = df[feature_2].value_counts().iloc[0:size].index.values
  
            for l in f1_levels: # create a copy for each unique value
                for m in f2_levels:
                    temp = sampled_rows.copy()

                    temp[feature_1] = l
                    temp['feature_1'] = feature_1

                    temp[feature_2] = m
                    temp['feature_2'] = feature_2
                    
                    scoring_data = scoring_data.append(temp)
        
        else:
            for l in f1_levels: # create a copy for each unique value
                temp = sampled_rows.copy()
                
                temp[feature_1] = l
                temp['feature'] = feature_1
                
                scoring_data = scoring_data.append(temp)

        return scoring_data.reset_index()

    
    def _score_data(self, df):
        """
        Send pandas df to DataRobot v2 API and return scores merged back with actuals

        Returns:
        --------
        pandas df
        """
        dataset = self.project.upload_dataset(df)
        pred_job = self.model.request_predictions(dataset.id)
        preds = pred_job.get_result_when_complete(max_wait=3600)
        
        preds = preds.merge(df.reset_index(drop=True), 
                            how='inner', 
                            left_index=True, 
                            right_index=True, 
                            validate='one_to_one')

        return preds

    
    def create_and_score_data(self, df, model, feature_1, feature_2=None):
        
        scoring_data = self._create_scoring_data(df, model, feature_1, feature_2)
        preds = self._score_data(scoring_data)
        preds.sort_values(by=feature_1, inplace=True)
        
        return preds 
    
    
    def get_features(self):
        raw = [feat_list for feat_list in self.project.get_featurelists()
               if feat_list.name == 'Informative Features'][0]
        raw_features = [
            {
                "name": feat,
                "type": dr.Feature.get(self.project.id, feat).feature_type
            }
            for feat in raw.features
        ]
        features = pd.DataFrame.from_dict(raw_features)

        to_keep = self.model.get_features_used()
        features = features.loc[features['name'].isin(to_keep),:]

        return features
    
    
    def get_feature_impact(self, n=None):
        feature_impact = [(f['featureName'],round(f['impactUnnormalized'],2)) for f in self.model.get_or_request_feature_impact()]
        
        if n is None:
            n = len(feature_impact)
        
        return feature_impact[0:n]
    
    
    def create_pdp_plot(self, 
                        feature_1, 
                        feature_2=None, 
                        max_bins=3, 
                        quantiles=False, 
                        normalize=False, 
                        include_ice=True,
                        n=200,
                        error_bars=False
        ):
        """
        Generate 1 or 2-way PDP plots
        
        Attributes:
        -----------
        
        feature_1: str
            Primary feature to be plotted on x-axis
        feature_2: (Optional) str
            Secondary feature used to segment the primary feature
        max_bins: int
            Maximum number of bins used to segment the secondary feature
        quantiles: boolean
            Set to True to segment the secondary feature using pd.qcut versus pd.cut
        normalize: boolean
            Re-index each PDP curve so they all start at the same value
        include_ice:
            Whether to plot ICE curves
        n: int
            Number of ICE curves to plot
        
        """
        
        assert isinstance(max_bins, int), 'max_bins must be an integer'
        assert 1 < max_bins <= 10, 'max_bins must be greater than 1 and less than or equal to 10'
        two_way_pdp = pd.DataFrame()
        
        #Don't re-score rows if we don't need to
#         if feature_1 in self.features:
#             preds = self.scores.loc[self.scores['feature']==feature_1, :].copy()
        
#         else:
            #Create scoring dataset and make predictions
        preds = self.create_and_score_data(self.df, self.model, feature_1, feature_2)
            
            #Keep track of the features already scored
#             self.scores = self.scores.append(preds)
#             self.features.append(feature_1)
        
        n = min(n, preds.shape[0])
        title = ''
        legend_title=''
        
        if self.target_type == 'Regression':
            col = 'prediction'
        else:
            if self.df[self.target].dtype in (np.int16, np.int32, np.int64, float):
                col = 'class_1.0'
            else:
                col = 'class_1'
    
        #Calculate one-way Partial Depedence
        if self.weights:
            w_mean = lambda x: np.average(x, weights=preds.loc[x.index, self.weights])
            w_std = lambda x: np.sqrt(np.cov(x, aweights=preds.loc[x.index, self.weights]))

            one_way_pdp = preds.groupby(feature_1).agg(mean = (col, w_mean),
                                                       std = (col, w_std),
                                                       count = (self.weights, 'sum')
                                                      ).reset_index().sort_values(by=feature_1, ascending=True)
        else:
            one_way_pdp = preds.groupby(feature_1)[col].agg(['mean','std','count']).reset_index().\
            sort_values(by=feature_1, ascending=True)
        
    
        #Calculate two-way Partial Depedence
        if feature_2:
            two_way_pdp = preds.copy()
            
            if feature_2 in self.df.select_dtypes(include=['number']).columns:
                if quantiles:
                    two_way_pdp[feature_2] = pd.qcut(two_way_pdp[feature_2], max_bins, duplicates='drop')
                else:
                    two_way_pdp[feature_2] = pd.cut(two_way_pdp[feature_2], max_bins, duplicates='drop')

            elif feature_2 in self.df.select_dtypes(include=['object']).columns:
                unique_values = len(two_way_pdp[feature_2].unique())
                if unique_values > max_bins:
                    top_bins = two_way_pdp[feature_2].value_counts()[0:max_bins-1].index.values
                    other_bin = two_way_pdp.loc[~two_way_pdp[feature_2].isin(top_bins), feature_2].unique()
                    
                    two_way_pdp[feature_2] = ['OTHER' if i in other_bin else i for i in two_way_pdp[feature_2]]
                else:
                    pass

            else:
                raise ValueError('feature_2 has to be either Numeric, Boolean, Categorical, Length, Percentage, or Currency')
            
            if self.weights:
                w_mean = lambda x: np.average(x, weights=two_way_pdp.loc[x.index, self.weights])
                w_std = lambda x: np.sqrt(np.cov(x, aweights=two_way_pdp.loc[x.index, self.weights]))

                two_way_pdp = two_way_pdp.groupby([feature_1, feature_2]).agg(mean = (col, w_mean),
                                                                        std = (col, w_std),
                                                                        count = (self.weights, 'sum')
                                                                        ).reset_index().sort_values(by=feature_1, ascending=True)
            else:
                two_way_pdp = two_way_pdp.groupby([feature_1, feature_2])[col].agg(['mean','std','count']).reset_index().\
                sort_values(by=feature_1, ascending=True)
        
        #Normalize scores
        if normalize:
            title = ' - Normalized'
            preds[col] = preds[col] - preds.groupby('index')[col].transform('first')
            one_way_pdp['mean'] = one_way_pdp['mean'] - one_way_pdp['mean'][0]
            
            if feature_2:
                two_way_pdp['mean'] = two_way_pdp['mean'] - two_way_pdp.groupby(feature_2)['mean'].transform('first')

        #Create plots
        fig = go.Figure()
        if include_ice:
            for i in preds['index'].unique()[0:n]:
                t = preds.loc[preds['index']==i,[feature_1, col]].copy()
                fig.add_trace(
                    go.Scatter(x=t[feature_1],y=t[col],
                              mode='lines',
                              showlegend=False,
                              line=dict(
                                  color='rgba(127, 127, 127,0.25)',
                                  width=0.5
                                      )
                              )
            )

        # Two-way PDP
        if feature_2:
            legend_title = feature_2
            for idx, f in enumerate(two_way_pdp[feature_2].unique()):
                temp = two_way_pdp.loc[two_way_pdp[feature_2]==f,:]
                fillcolor = None
                fill=None

                if error_bars:
                    fillcolor='rgba(20, 20, 20, 0.3)'
                    fill='tonexty'
                    fig.add_trace(
                        go.Scatter(
                            name=f'Lower Bound {f}',
                            showlegend=False,
                            x=temp[feature_1],
                            y=temp['mean']-temp['std']/np.sqrt(temp['count']),
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0)
                        )
                    )

                fig.add_trace(go.Scatter(x=temp[feature_1], y=temp['mean'],
                            mode='lines+markers',
                            name=str(f),
                            marker=dict(size=4,
                                       color=self.colors[idx]),
                            line=dict(
                                color=self.colors[idx],
                                width=2
                            ),
                            fillcolor=self.opacities[idx],
                            fill=fill,
                            #connectgaps=True
                                        )
                             )

                if error_bars:
                    fig.add_trace(
                        go.Scatter(
                            name=f'Upper Bound {f}',
                            showlegend=False,
                            x=temp[feature_1],
                            y=temp['mean']+temp['std']/np.sqrt(temp['count']),
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            fillcolor=self.opacities[idx],
                            fill=fill
                        )
                    )

        else:
            fillcolor = None
            fill=None

            if error_bars:
                fillcolor='rgba(20, 20, 20, 0.1)'
                fill='tonexty'
                
                fig.add_trace(
                    go.Scatter(
                        name='Lower Bound',
                        showlegend=False,
                        x=one_way_pdp[feature_1],
                        y=one_way_pdp['mean']-one_way_pdp['std']/np.sqrt(one_way_pdp['count']),
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0)
                    )
                )

            fig.add_trace(
                        go.Scatter(
                                    x=one_way_pdp[feature_1], 
                                    y=one_way_pdp['mean'],
                                    mode='lines+markers',
                                    name='Partial Dependence',
                                    marker=dict(size=6,
                                               color='black'
                                               ),
                                    line=dict(
                                        color='black',
                                        width=4
                                    ),
                                    fillcolor=fillcolor,
                                    fill=fill
                                                )
                                     )
            
            if error_bars:
                fig.add_trace(
                    go.Scatter(
                        name='Upper Bound',
                        showlegend=False,
                        x=one_way_pdp[feature_1],
                        y=one_way_pdp['mean']+one_way_pdp['std']/np.sqrt(one_way_pdp['count']),
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        fillcolor=fillcolor,
                        fill=fill
                    )
                )


        fig.update_layout(title={
                        'text': f'<b>Partial Dependence</b>{title}',
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                          },
                  legend_title=f'{legend_title}'
                 )
        fig.update_yaxes(title=self.target)
        fig.update_xaxes(title=feature_1)
        fig.show()
        
        return preds, one_way_pdp, two_way_pdp