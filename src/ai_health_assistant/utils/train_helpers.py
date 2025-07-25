import pandas as pd
import numpy as np
import sys
if sys.prefix != sys.base_prefix:
    import matplotlib
    matplotlib.use("Agg") 

import matplotlib.pyplot as plt


from sklearn.model_selection import  GridSearchCV, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

from pathlib import Path
import joblib
import os


############## Entrenament del model ###############################
def train_models(X_train, y_train, X_test, y_test, pipeline, param_grid, scoring = 'f1', cv = 'StratifiedKFold', n_iter = 100, search_type = 'random'):
    '''
    Entrenament del model amb cerca d'hiperparàmetres.
    
    Arguments:
    - X_train: característiques del conjunt d'entrenament
    - y_train: variable objectiu del conjunt d'entrenament
    - X_test: característiques del conjunt de test
    - y_test: variable objectiu del conjunt de test
    - pipeline: pipeline amb almenys el preprocessador i el classificador
    - param_grid: paràmetres per al gridsearch (si és un diccionari amb diversos classificadors es pot aplicar com: param_grid[nom_model])
    - scoring: per defecte F1 per a la classe 1
    - cv: validació creuada; per defecte StratifiedKFold amb 5 splits
    - n_iter: nombre d'iteracions; per defecte 100
    - search_type: 'random' (per defecte) o 'grid' per a GridSearchCV
    
    Retorna:
    - best_est, y_train_pred, train_report, y_test_pred, test_report, best_params, best_score
    '''
    if scoring == 'f1':
        scoring = make_scorer(f1_score, pos_label=1)
    if cv == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if search_type == 'grid':
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            refit=True
        )
    
    print(f"Entrenant model...")
    search.fit(X_train, y_train)
    
    # Millor estimador
    best_est = search.best_estimator_

    # Predicció sobre el conjunt d'entrenament
    y_train_pred = best_est.predict(X_train)

    train_report = classification_report(
        y_train,
        y_train_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    # Predicció sobre el conjunt de test
    y_test_pred = best_est.predict(X_test)

    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    # Print del resum amb els valors més importants
    print(f"\nTrain F1 (1): {train_report["1"]["f1-score"]:.4f} | Test F1 (1): {test_report["1"]["f1-score"]:.4f} | Train Acc: {train_report["accuracy"]:.4f} | Test Acc: {test_report["accuracy"]:.4f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    return best_est, y_train_pred, train_report, y_test_pred, test_report,  search.best_params_, search.best_score_


############## Append Results ###############################
def append_results (list_results, model_name, train_report, test_report, best_params, best_score, experiment = None):
    '''
    Crea un **dataframe amb els resultats** de la predicció i del model, fa un append a una llista,
    i retorna el dataframe. Desa el CSV a results.
    Les columnes que es poden mostrar són:
    - "Model"
    - "Experiment"
    - "Best Params"
    - "Best CV"
    - "Train F1 (1)"
    - "Train F1 (macro global)"
    - "Train Accuracy"
    - "Test Precision (1)"
    - "Test Recall (1)"
    - "Test F1 (1)"
    - "Test F1 (macro global)"
    - "Test Accuracy"

    Arguments:
    - list_results: llista on guardem els resultats
    - model_name: nom del model
    - train_report: informe de la predicció sobre el conjunt d'entrenament
    - test_report: informe de la predicció sobre el conjunt de test
    - best_params: millors paràmetres trobats
    - best_score: millor resultat de CV trobat
    - experiment: nom de l'experiment (opcional)

    Retorna:
    - results_df: dataframe amb els resultats
    '''
    if experiment is None:
        experiment = np.nan

    list_results.append({
        "Model":                 model_name,
        "Experiment":            f"{model_name}_{experiment}", # En cas de registrar un experiment
        
        "Best Params":           best_params,
        "Best CV":               best_score,

        "Train F1 (1)":          train_report["1"]["f1-score"],
        "Train F1 (macro global)": train_report["macro avg"]["f1-score"],
        "Train Accuracy":        train_report["accuracy"],

        "Test Precision (1)":    test_report["1"]["precision"],
        "Test Recall (1)":       test_report["1"]["recall"],
        "Test F1 (1)":           test_report["1"]["f1-score"],
        "Test F1 (macro global)": test_report["macro avg"]["f1-score"],
        "Test Accuracy":         test_report["accuracy"],
    })

    results_df = pd.DataFrame(list_results).round(5)

    return results_df

############## Corba d'aprenentatge ###############################
def plot_learning_curve(model_name, best_est, X_train, y_train, save = 'no', score = 'f1'):
    '''
    Genera una corba d'aprenentatge per veure com el model aprèn sobre el conjunt d'entrenament.
    
    Arguments:
    - model_name: nom del model
    - best_est: millor estimador trobat
    - X: característiques
    - y: variable objectiu
    - save: per defecte 'no' (mostra la gràfica en notebooks); si és 'yes', desa la gràfica (i no la mostra)
    - score: per defecte 'f1'; si no, indicar la mètrica a utilitzar
    '''
    if score == 'f1':
        scorer = make_scorer(f1_score, pos_label=1)
    else:
        scorer = score

    train_sizes, train_scores, val_scores = learning_curve(
        best_est, X_train, y_train,
        cv=5,
        scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True, 
        random_state=42,
        verbose=0
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    title = f"Corba d'aprenentatge - {model_name}"

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Train')
    plt.plot(train_sizes, val_mean,   'o-', label='CV')
    plt.title(title)
    plt.xlabel('Grandària del conjunt')
    plt.ylabel(f'Score ({score})')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    if save.lower() == 'yes':
        fname = f"lc_{model_name}.png"
        out_path = f"results/03_training/{fname}"
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Corba d'aprenentatge guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()

############## Matriu de confusió ###############################
def mat_confusio(title_name, y_true, y_pred, save = 'no'):
    '''
    Matriu de confusió sobre el conjunt de test.
    Desa la matriu de confusió al directori results/03_training
    '''
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1] )
    disp.plot(cmap='Blues')
    plt.title(f"Matriu de confusió - {title_name}")
    
    if save.lower() == 'yes':
        fname = f"cm_{title_name}.png"
        out_path = f'results/03_training/{fname}'
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Confusion matrix guardada a: {out_path}")
        plt.close()
        return
    
    # Si no, mostrem la gràfica
    plt.show()

############### Registre Mètriques ###############################
def update_metrics_file(metrics: pd.DataFrame, filename="results/03_training/metrics.csv"):
    columnas = ["Model", "Train F1 (1)", "Train F1 (macro global)", "Train Accuracy", "Test Precision (1)", "Test Recall (1)", "Test F1 (1)", "Test F1 (macro global)", "Test Accuracy", "Best Params"]

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=columnas)
    
    fila_nova = metrics[columnas].copy()
    model_name = metrics["Model"].iloc[0]
    rewrite = df["Model"] == model_name
    
    if rewrite.any():
        df.loc[rewrite, columnas] = fila_nova.values
    else:
        df = pd.concat([df, fila_nova], ignore_index=True)
    
    df = df.sort_values(by="Test F1 (1)", ascending=False)
    
    df.to_csv(filename, index=False)
    print(f'\nMètriques guardades a {filename}\n')


def update_experiments_file(metrics: pd.DataFrame, filename="../results/02_experiments/experiments.csv"):
    columnas = ["Experiment", "Train F1 (1)", "Train F1 (macro global)", "Train Accuracy", "Test Precision (1)", "Test Recall (1)", "Test F1 (1)", "Test F1 (macro global)", "Test Accuracy", "Best Params"]

    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=columnas)

    for _, fila_nova in metrics[columnas].iterrows():
        experiment_name = fila_nova["Experiment"]
        rewrite = df["Experiment"] == experiment_name
        if rewrite.any():
            for col in columnas:
                df.loc[rewrite, col] = fila_nova[col]
        else:
            df = pd.concat([df, pd.DataFrame([fila_nova])], ignore_index=True)

    df = df.sort_values(by="Test F1 (1)", ascending=False)

    df.to_csv(filename, index=False)
    print(f'\nMètriques guardades a {filename}\n')

############### Guardem Models ###############################
def save_model(best_estimator, model_name, save_external='no'):
    """
    Si save_external = 'yes', desa el model en:
      1) ./models/{model_name}_model.joblib
      2) ../AI-Health-Assistant-WebApp/backend/models/{model_name}_model.joblib
    Per tal de poder-lo utilitzar en la webapp.

    En cas que save_external = 'no', només desa el model en:
      1) ./models/{model_name}_model.joblib
    """
    # RUTA DINS DEL REPOSITORI ACTUAL
    local_dir = Path(__file__).parent.parent.parent.parent / "models"
    local_path = local_dir / f"{model_name}_model.joblib"

    joblib.dump(best_estimator, local_path)  # Desa el model localment
    print(f"\nModel desat localment a: {local_path}\n")

    # RUTA EXTERNA A LA WEBAPP
    if save_external.lower() == 'yes':
        # La ruta és relativa a la ubicació de les meves carpetes
        external_dir = (Path(__file__).parent.parent.parent.parent.parent / "AI-Health-Assistant-WebApp" / "backend" / "models")
        external_path = external_dir / f"{model_name}_model.joblib"

        joblib.dump(best_estimator, external_path)  # Desa el model externament, a la webapp
        print(f"\nModel desat externament a: {external_path}\n")
