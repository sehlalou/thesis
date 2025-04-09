import datetime
from pathlib import Path
import time
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from tqdm import tqdm

from preprocess import clean_signal
import config as cfg
import config_trans as hp 
from model_transformer import VisionTransformer, ViTModelConfig, CNN_ViT_Hybrid

# Fixe la seed et active TF32 si disponible
torch.manual_seed(cfg.RANDOM_SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DetectionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]
            ecg_data = f[key][dw.start_index:dw.end_index, 0]
        
        ecg_data = clean_signal(ecg_data)

        ecg_data = torch.tensor(ecg_data.copy(), dtype=torch.float32)
        # Pour des données à un seul canal, ajouter une dimension de canal
        ecg_data = ecg_data.unsqueeze(0)
        label = torch.tensor(dw.label, dtype=torch.long)
        return ecg_data, label


def load_dataset():
    """
    Charge le dataset entier depuis le fichier CSV.
    """
    if cfg.DETECTION:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
        print("Tâche de détection")
    else:
        dataset_path = Path(hp.DATASET_PATH, f"dataset_identification_ecg_{cfg.WINDOW_SIZE}_{cfg.LOOK_A_HEAD}.csv")
        print("Tâche d'identification")
    df = pd.read_csv(dataset_path)
    
    return df


def create_train_test_split(test_size=0.2):
    """
    Divise le dataset en train set et test set en s'assurant que tous les enregistrements d'un patient restent ensemble.
    """
    df = load_dataset()
   
    # Calcul du label représentatif par patient
    patient_labels = df.groupby("patient_id")["label"].agg(lambda x: x.mode()[0]).reset_index()
    groups = patient_labels["patient_id"].values
    labels = patient_labels["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.RANDOM_SEED)
    train_idx, test_idx = next(gss.split(patient_labels, groups=groups, y=labels))
    train_patients = patient_labels.loc[train_idx, "patient_id"].values
    test_patients = patient_labels.loc[test_idx, "patient_id"].values

    train_df = df[df["patient_id"].isin(train_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]
    
    print(f"Split: {len(train_patients)} patients dans le train set, {len(test_patients)} patients dans le test set")
    return train_df, test_df


def create_k_fold_splits(train_df, n_splits=5):
    """
    Crée les splits K-fold sur le train set en se basant sur les patients.
    """
    # Calcul du label représentatif par patient pour le train set
    patient_labels = train_df.groupby("patient_id")["label"].agg(lambda x: x.mode()[0]).reset_index()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(patient_labels["patient_id"],
                                                                patient_labels["label"],
                                                                groups=patient_labels["patient_id"])):
        # Récupérer les identifiants patients pour les splits train et validation
        train_patients = patient_labels.loc[train_idx, "patient_id"].values
        val_patients = patient_labels.loc[val_idx, "patient_id"].values
        
        fold_train_df = train_df[train_df["patient_id"].isin(train_patients)]
        fold_val_df = train_df[train_df["patient_id"].isin(val_patients)]
        
        train_dataset = DetectionDataset(fold_train_df)
        val_dataset = DetectionDataset(fold_val_df)
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.NUM_PROC_WORKERS,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=cfg.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.NUM_PROC_WORKERS,
                                pin_memory=True)
        
        folds.append({
            "fold": fold_idx,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "train_patients": train_patients,
            "val_patients": val_patients
        })
        
        print(f"Fold {fold_idx}: {len(train_patients)} patients train, {len(val_patients)} patients val")
        
    return folds


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def configure_optimizers(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    return optimizer


@torch.no_grad()
def estimate_loss(model, device, data_loader, criterion):
    model.eval()
    losses = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device).long()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def estimate_metrics(model, data_loader, device):
    """
    Retourne un dictionnaire contenant :
    - roc_auc
    - accuracy
    - sensitivity
    - specificity
    - f1_score
    """
    model.eval()
    list_y_true = []
    list_y_pred = []
    list_y_pred_prob = []  # Probabilité pour la classe positive (index 1)
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device).long()
        list_y_true.extend(y.cpu().numpy())
        y_pred = model(x)
        preds = torch.argmax(y_pred, dim=1)
        list_y_pred.extend(preds.cpu().numpy())
        prob_class1 = y_pred[:, 1].cpu().numpy()
        list_y_pred_prob.extend(prob_class1)

    roc_auc = roc_auc_score(list_y_true, list_y_pred_prob)
    cm = confusion_matrix(list_y_true, list_y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score
    }


def print_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time // 60
    elapsed_seconds = elapsed_time % 60
    print(f"Temps total d'entraînement : {int(elapsed_minutes)} minutes et {int(elapsed_seconds)} secondes")


def train_fold(fold_data, fold_num):
    print(f"\nEntraînement du fold {fold_num}")
    device = get_device()
    # Configuration du modèle Transformer
    config = ViTModelConfig(
        input_size=cfg.WINDOW_SIZE,
        patch_size=hp.PATCH_SIZE,
        emb_dim=hp.EMB_DIM,
        num_layers=hp.NUM_LAYERS,
        num_heads=hp.NUM_HEADS,
        mlp_dim=hp.MLP_DIM,
        num_classes=2,
        dropout_rate=hp.DROPOUT_RATE
    )
    model = VisionTransformer(config)
    model = model.to(device)
    optimizer = configure_optimizers(model)
    criterion = nn.CrossEntropyLoss() 

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_metrics = []
    
    for epoch in range(cfg.EPOCH):
        model.train()
        train_losses = []
        for x, y in tqdm(fold_data["train_loader"], desc=f"Fold {fold_num} Epoch {epoch+1}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_losses)
        # Calcul des métriques sur le train set
        train_metrics = estimate_metrics(model, fold_data["train_loader"], device)
        val_loss = estimate_loss(model, device, fold_data["val_loader"], criterion)
        val_metrics = estimate_metrics(model, fold_data["val_loader"], device)
        print(f"Fold {fold_num} Epoch {epoch + 1} : train loss {train_loss:.4f}, train accuracy {train_metrics['accuracy']:.4f}, train roc_auc {train_metrics['roc_auc']:.4f}")
        print(f"Fold {fold_num} Epoch {epoch + 1} : val loss {val_loss:.4f}, val accuracy {val_metrics['accuracy']:.4f}, val roc_auc {val_metrics['roc_auc']:.4f}, "
              f"sensitivity {val_metrics['sensitivity']:.4f}, specificity {val_metrics['specificity']:.4f}")

        epoch_metrics.append({
            "fold": fold_num,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"]
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= cfg.PATIENCE:
            print(f"Early stopping dans le fold {fold_num} à l'epoch {epoch + 1}")
            model.load_state_dict(best_model_state)
            break

    return model, epoch_metrics


def train_model_cv_and_final():
    start_time = time.time()
    # 1. Division du dataset en train et test
    train_df, test_df = create_train_test_split(test_size=0.2)
    
    # 2. Cross Validation sur le train set
    folds = create_k_fold_splits(train_df, n_splits=cfg.K_FOLDS)  # cfg.K_FOLDS doit être défini dans le fichier de config
    all_epoch_metrics = []
    cv_metrics = []  # Pour stocker les métriques finales de chaque fold

    for fold_data in folds:
        fold_num = fold_data["fold"]
        best_model, epoch_metrics = train_fold(fold_data, fold_num)
        all_epoch_metrics.extend(epoch_metrics)
        
        final_cv_metrics = estimate_metrics(best_model, fold_data["val_loader"], get_device())
        print(f"Fold {fold_num} métriques finales:")
        print(f"  Val Accuracy: {final_cv_metrics['accuracy']:.4f}")
        print(f"  Val ROC AUC: {final_cv_metrics['roc_auc']:.4f}")
        print(f"  Val Sensitivity: {final_cv_metrics['sensitivity']:.4f}")
        print(f"  Val Specificity: {final_cv_metrics['specificity']:.4f}")
        cv_metrics.append(final_cv_metrics)
        
        # Sauvegarder le modèle pour ce fold
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = Path(hp.LOG_DL_PATH, f"fold_{fold_num}_{timestamp}")
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(best_model.state_dict(), folder / "model.pt")
        pd.DataFrame(epoch_metrics).to_csv(folder / "epoch_metrics.csv", index=False)

    # Calculer la moyenne et l'écart-type sur les folds (accuracy, roc_auc, sensitivity et specificity)
    accs = [m["accuracy"] for m in cv_metrics]
    roc_aucs = [m["roc_auc"] for m in cv_metrics]
    sensitivities = [m["sensitivity"] for m in cv_metrics]
    specificities = [m["specificity"] for m in cv_metrics]

    print("\n--- Résultats Cross Validation ---")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"Specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")

    # 3. Entraîner le modèle final sur l'ensemble complet du train set et évaluer sur le test set
    print("\nEntraînement du modèle final sur l'ensemble complet du train set...")
    train_dataset = DetectionDataset(train_df)
    test_dataset = DetectionDataset(test_df)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_PROC_WORKERS,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.NUM_PROC_WORKERS,
                             pin_memory=True)

    device = get_device()
    config = ViTModelConfig(
        input_size=cfg.WINDOW_SIZE,
        patch_size=hp.PATCH_SIZE,
        emb_dim=hp.EMB_DIM,
        num_layers=hp.NUM_LAYERS,
        num_heads=hp.NUM_HEADS,
        mlp_dim=hp.MLP_DIM,
        num_classes=2,
        dropout_rate=hp.DROPOUT_RATE
    )
    final_model = VisionTransformer(config)
    final_model = final_model.to(device)
    optimizer = configure_optimizers(final_model)
    criterion = nn.CrossEntropyLoss()

    # Entraînement sur l'ensemble complet du train set sans validation intermédiaire et early stopping
    for epoch in range(cfg.EPOCH):
        final_model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Final Model Epoch {epoch+1}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = final_model(x)
            loss = criterion(y_pred, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_losses)
        train_metrics = estimate_metrics(final_model, train_loader, device)
        print(f"Final Model Epoch {epoch + 1} : train loss {train_loss:.4f}, train accuracy {train_metrics['accuracy']:.4f}, train AUROC {train_metrics['roc_auc']:.4f}")

    # Évaluation finale sur le test set
    test_loss = estimate_loss(final_model, device, test_loader, criterion)
    test_metrics = estimate_metrics(final_model, test_loader, device)
    print("\n--- Évaluation sur le Test Set ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")

    total_training_time = time.time() - start_time
    print_elapsed_time(start_time)
    
    # Sauvegarder les métriques globales (CV + test)
    all_results = {
        "cv_accuracy_mean": np.mean(accs),
        "cv_accuracy_std": np.std(accs),
        "cv_roc_auc_mean": np.mean(roc_aucs),
        "cv_roc_auc_std": np.std(roc_aucs),
        "cv_sensitivity_mean": np.mean(sensitivities),
        "cv_sensitivity_std": np.std(sensitivities),
        "cv_specificity_mean": np.mean(specificities),
        "cv_specificity_std": np.std(specificities),
        "test_loss": test_loss,
        "test_accuracy": test_metrics['accuracy'],
        "test_roc_auc": test_metrics['roc_auc'],
        "test_sensitivity": test_metrics['sensitivity'],
        "test_specificity": test_metrics['specificity']
    }
    pd.DataFrame([all_results]).to_csv(Path(hp.LOG_DL_PATH, "final_results_study_vit.csv"), index=False)


if __name__ == "__main__":
    train_model_cv_and_final()
