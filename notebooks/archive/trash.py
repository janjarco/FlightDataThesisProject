
# %% Uplift modelling from scikit-uplift
# -------------------------------------
model_names = ["DoubleMlPLR1","DoubleMLPLR2","DoubleMLIRM1","DoubleMLIRM2", "SoloModel", "ClassTransform", "TwoModelTrmnt", "TwoModelCtrl"]
def uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test, 
                    classifier, classifier_params, regressor, regressor_params, 
                    selected_features, y_col, d_col):
    # declare sample rfoost regressor with sample parameters
    # add time measurement 
    from datetime import datetime
    start_time = datetime.now()

    models_results = {
        'approach': [],
        'uplift': []
    }
    # hide warnings
    import warnings
    warnings.filterwarnings('ignore')


    # Double ML
    # fit the DoubleMLPLR model

    from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # split df into training and test sets
    df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=selected_features, y_col=y_col, d_cols=d_col)

    # #  DoubleML 1
    dml_plr_1 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                          ml_m=classifier(**classifier_params),
                        #   ml_m=regressor(**regressor_params),  
                          dml_procedure='dml1',
                          n_folds=5
                          )
    dml_plr_1 = dml_plr_1.fit(store_predictions = True, store_models = True)
    dml_plr_1_score = dml_plr_1.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR1')
    models_results['uplift'].append(dml_plr_1_score)
    # DOuble ML 2
    dml_plr_2 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                        #   ml_m=classifier(**classifier_params),
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml2',
                          n_folds=5
                          )
    dml_plr_2=dml_plr_2.fit(store_predictions = True, store_models = True)
    dml_plr_2_score = dml_plr_2.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR2')
    models_results['uplift'].append(dml_plr_2_score)

    #  DoubleML 1
    dml_irm_1 = DoubleMLIRM(df_doubleml, 
                          ml_g=classifier(**classifier_params), 
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml1',
                          n_folds=5
                          )
    dml_irm_1 = dml_irm_1.fit(store_predictions = True, store_models = True)
    dml_irm_1_score = dml_irm_1.summary['coef'].values[0]

    models_results['approach'].append('DoubleMLIRM1')
    models_results['uplift'].append(dml_irm_1_score)
    # DOuble ML 2
    dml_irm_2 = DoubleMLIRM(df_doubleml, 
                          ml_g=classifier(**classifier_params), 
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml2',
                          n_folds=5
                          )
    dml_irm_2=dml_irm_2.fit(store_predictions = True, store_models = True)
    dml_irm_2_score = dml_irm_2.summary['coef'].values[0]

    models_results['approach'].append('DoubleML2')
    models_results['uplift'].append(dml_irm_2_score)
    # print("Double ML done")

    # Solo Model
    from sklift.metrics import uplift_at_k, weighted_average_uplift
    from sklift.viz import plot_uplift_preds
    from sklift.models import SoloModel

    sm = SoloModel(classifier(**classifier_params))
    sm = sm.fit(X_train, y_train, treat_train)

    uplift_sm = sm.predict(X_test)

    # sm_score = uplift_at_k(y_true=y_test, uplift=uplift_sm, treatment=treat_test, strategy=strategy, k=uplift_k)
    sm_score = weighted_average_uplift(y_true=y_test, uplift=uplift_sm, treatment=treat_test, strategy='overall')

    models_results['approach'].append('SoloModel')
    models_results['uplift'].append(sm_score)
    # print("Solo Model done")
    # get conditional probabilities (predictions) of performing the target action 
    # during interaction for each object
    # sm_trmnt_preds = sm.trmnt_preds_
    # And conditional probabilities (predictions) of performing the target action 
    # without interaction for each object
    # sm_ctrl_preds = sm.ctrl_preds_

    from sklift.models import ClassTransformation

    ct = ClassTransformation(classifier(**classifier_params))
    ct = ct.fit(X_train, y_train, treat_train)

    uplift_ct = ct.predict(X_test)

    ct_score = uplift_at_k(y_true=y_test, uplift=uplift_ct, treatment=treat_test, strategy='by_group', k=.5)
    # ct_score = weighted_average_uplift(y_true=y_test, uplift=uplift_ct, treatment=treat_test, strategy='overall')

    # plot_uplift_preds(trmnt_preds=ct.trmnt_preds_, ctrl_preds=ct.ctrl_preds_)

    models_results['approach'].append('ClassTransformation')
    models_results['uplift'].append(ct_score)
    # print("Class Transformation done")
    # Two models
    import importlib
    import sklift.models
    importlib.reload(sklift.models)


    import sklift.models.models
    importlib.reload(sklift.models.models)
    from sklift.models import TwoModels
    # Two models treatment

    tm_trmnt = TwoModels(
        estimator_trmnt=classifier(**classifier_params), 
        estimator_ctrl=classifier(**classifier_params), 
        method='ddr_treatment'
    )
    tm_trmnt = tm_trmnt.fit(X_train, y_train, treat_train)

    uplift_tm_trmnt = tm_trmnt.predict(X_test)

    # tm_trmnt_score = uplift_at_k(y_true=y_test, uplift=uplift_tm_trmnt, treatment=treat_test, strategy='by_group', k=.5)
    tm_trmnt_score = weighted_average_uplift(y_true=y_test, uplift=uplift_tm_trmnt, treatment=treat_test, strategy='overall')
    models_results['approach'].append('TwoModels_ddr_treatment')
    models_results['uplift'].append(tm_trmnt_score)
    
    # plot_uplift_preds(trmnt_preds=tm_trmnt.trmnt_preds_, ctrl_preds=tm_trmnt.ctrl_preds_)

    # Two models control
    tm_ctrl = TwoModels(
        estimator_trmnt=classifier(**classifier_params), 
        estimator_ctrl=classifier(**classifier_params), 
        method='ddr_control'
    )
    tm_ctrl = tm_ctrl.fit(X_train, y_train, treat_train)

    uplift_tm_ctrl = tm_ctrl.predict(X_test)

    # tm_ctrl_score = uplift_at_k(y_true=y_test, uplift=uplift_tm_ctrl, treatment=treat_test, strategy='by_group', k=.5)
    tm_ctrl_score = weighted_average_uplift(y_true=y_test, uplift=uplift_tm_ctrl, treatment=treat_test, strategy='overall')

    models_results['approach'].append('TwoModels_ddr_control')
    models_results['uplift'].append(tm_ctrl_score)
    # print("Two Models done")

    # plot_uplift_preds(trmnt_preds=tm_ctrl.trmnt_preds_, ctrl_preds=tm_ctrl.ctrl_preds_)
    # print timedifference
    print('Time taken: {}'.format(datetime.now() - start_time), )

    return pd.DataFrame(models_results), dict(zip(model_names, [dml_plr_1, dml_plr_2, dml_irm_1, dml_irm_2, sm, ct, tm_trmnt, tm_ctrl]))



 # %% 
#  implementation with casualml package that doesnot work for this causal environment
 # --------------------
model_names = ["IPW", "OR", "DoublyRobust", "DoubleMlPLR1", "DoubleMLPLR2", "DoubleMLIRM1", "DoubleMLIRM2"]
def ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test, 
                    classifier, classifier_params, regressor, regressor_params, 
                    selected_features, y_col, d_col):
    # declare sample rfoost regressor with sample parameters
    # add time measurement 
    from datetime import datetime
    start_time = datetime.now()

    models_results = {
        'approach': [],
        'uplift': []
    }
    # hide warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Inverse Probability Weighting (IPW)
    from causalml.inference.meta import IPW
    ipw = IPW(learner=classifier(**classifier_params))
    ipw.fit(X_train, treat_train, y_train)
    ipw_score = ipw.estimate_ate(X_test, treat_test, y_test)[0][0]

    models_results['approach'].append('IPW')
    models_results['uplift'].append(ipw_score)

    # Outcome Regression (OR)
    from causalml.inference.meta import BaseRRegressor
    or_model = BaseRRegressor(learner=regressor(**regressor_params))
    or_model.fit(X_train, treat_train, y_train)
    or_score = or_model.estimate_ate(X_test, treat_test, y_test)[0][0]

    models_results['approach'].append('OR')
    models_results['uplift'].append(or_score)

    # Doubly Robust (DR)
    from causalml.inference.meta import BaseDRRegressor
    dr_model = BaseDRRegressor(learner=regressor(**regressor_params),
                                propensity_learner=classifier(**classifier_params))
    dr_model.fit(X_train, treat_train, y_train)
    dr_score = dr_model.estimate_ate(X_test, treat_test, y_test)[0][0]

    models_results['approach'].append('DoublyRobust')
    models_results['uplift'].append(dr_score)

    from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # split df into training and test sets
    df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=selected_features, y_col=y_col, d_cols=d_col)

    # #  DoubleML 1
    dml_plr_1 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                          ml_m=classifier(**classifier_params),
                        #   ml_m=regressor(**regressor_params),  
                          dml_procedure='dml1',
                          n_folds=5
                          )
    dml_plr_1 = dml_plr_1.fit(store_predictions = True, store_models = True)
    dml_plr_1_score = dml_plr_1.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR1')
    models_results['uplift'].append(dml_plr_1_score)
    # DOuble ML 2
    dml_plr_2 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                        #   ml_m=classifier(**classifier_params),
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml2',
                          n_folds=5
                          )
    dml_plr_2=dml_plr_2.fit(store_predictions = True, store_models = True)
    dml_plr_2_score = dml_plr_2.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR2')
    models_results['uplift'].append(dml_plr_2_score)

    #  DoubleML 1
    dml_irm_1 = DoubleMLIRM(df_doubleml, 
                          ml_g=classifier(**classifier_params), 
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml1',
                          n_folds=5
                          )
    dml_irm_1 = dml_irm_1.fit(store_predictions = True, store_models = True)
    dml_irm_1_score = dml_irm_1.summary['coef'].values[0]

    models_results['approach'].append('DoubleMLIRM1')
    models_results['uplift'].append(dml_irm_1_score)
    # DOuble ML 2
    dml_irm_2 = DoubleMLIRM(df_doubleml, 
                          ml_g=classifier(**classifier_params), 
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml2',
                          n_folds=5
                          )
    dml_irm_2=dml_irm_2.fit(store_predictions = True, store_models = True)
    dml_irm_2_score = dml_irm_2.summary['coef'].values[0]

    models_results['approach'].append('DoubleML2')
    models_results['uplift'].append(dml_irm_2_score)
    # print("Double ML done")

    

    # plot_uplift_preds(trmnt_preds=tm_ctrl.trmnt_preds_, ctrl_preds=tm_ctrl.ctrl_preds_)
    # print timedifference
    print('Time taken: {}'.format(datetime.now() - start_time), )

    return pd.DataFrame(models_results), dict(zip(model_names, [ipw, or_model, dr_model, dml_plr_1, dml_plr_2, dml_irm_1, dml_irm_2]))
