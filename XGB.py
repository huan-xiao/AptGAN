import torch
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import sys

seed_num=0
np.random.seed(seed_num)
np.set_printoptions(threshold=sys.maxsize)


def print_confusion_matrix(model_name, y_test, predictions):
    
    class_names = [0,1]
    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure(1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N','P'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./results/'+model_name+'_Confusion_Matrix.png')
    
    
def print_metrics(model_name, y_test, y_pred):
    
    predictions = np.round(y_pred) # threhold=0.5
    print_confusion_matrix(model_name, y_test, predictions)
    #print(classification_report(y_test, predictions,digits=3))
    print('AUC:', end=' ')
    print(round(roc_auc_score(y_test, y_pred),3))
    print('Accuracy:', end=' ')
    print(round(accuracy_score(y_test, predictions),3))
    print('Precision:', end=' ')
    print(round(precision_score(y_test, predictions),3))
    print('Recall:', end=' ')
    print(round(recall_score(y_test, predictions),3))
    print('F1-Score:', end=' ')
    print(round(f1_score(y_test, predictions),3))
    print('MCC:', end=' ')
    print(round(matthews_corrcoef(y_test, predictions),3))
    print('Kappa:', end=' ')
    print(round(cohen_kappa_score(y_test, predictions),3))

    
if __name__ == "__main__":
    
    x_train = torch.load("./dataset/training/x_train.pt") # [5888, 4775]
    x_test = torch.load("./dataset/training/x_test.pt") # [580, 4775]
    y_train = torch.load("./dataset/training/y_train.pt") # [5888]
    y_test = torch.load("./dataset/training/y_test.pt") # [580]
    
    x_train = x_train.detach().numpy()
    x_test = x_test.detach().numpy()
    y_train = y_train.detach().numpy()
    y_test = y_test.detach().numpy()
    
    clf = XGBClassifier(objective='binary:logistic', n_estimators=941, eta=0.020070367322859408, max_depth=21, min_child_weight=5, colsample_bytree=0.12930245487292172, subsample=0.8457886927155607, reg_lambda=2, gamma=0, alpha=0, scale_pos_weight=2, seed=seed_num, booster='gbtree')
    
    evalset = [(x_train, y_train), (x_test,y_test)]
    clf.fit(x_train, y_train, eval_set=evalset)
    results = clf.evals_result()
    
    # plot learning curves
    plt.figure(0)
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig("./results/XGB_loss_curve.png")
    
    clf.save_model("./models/XGB_classifier.json")
    
    y_pred = clf.predict_proba(x_test)[:,1]
    
    print_metrics("XGB", y_test, y_pred)
    
    
    