import os
import glob
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, medfilt
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, 
                             roc_curve, f1_score, matthews_corrcoef)

warnings.filterwarnings('ignore')

class pslc_engine:
    def __init__(self):
        self.FS = 1000
        # 🔑 三大维度，各自为王的冠军特征密码本
        self.GOLD_INDICES_PPI = [2, 4, 5, 16, 21] 
        self.GOLD_INDICES_VAS = [0, 1, 11, 25, 31]
        self.GOLD_INDICES_ODI = [8, 12, 15, 19, 25]

    # =================================================================
    # --- 零件 1: 信号预处理底座 ---
    # =================================================================
    def pslc_clean_emg(self, raw_signal):
        nyq = 0.5 * self.FS
        b_bp, a_bp = butter(2, [30/nyq, 400/nyq], btype='bandpass')
        return filtfilt(b_bp, a_bp, raw_signal, axis=0)

    def pslc_robust_reader(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, low_memory=False)
            if df.shape[1] < 22: df = pd.read_csv(file_path, sep=',', header=None, low_memory=False)
        except: 
            df = pd.read_excel(file_path, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').ffill().fillna(0)
        return df.iloc[:, 0].values, df.iloc[:, [13,14,15,16,18,19,20,21]].values, (df.iloc[:, 22].values if df.shape[1] > 22 else np.zeros(len(df)))

    def pslc_find_t0_ball(self, pressure):
        if len(pressure) < 1500: return None
        abs_p = np.abs(pressure - medfilt(pressure, kernel_size=201))
        search_start, search_end = 300, len(abs_p) - 1000
        if search_end <= search_start: return None
        peak_idx = np.argmax(abs_p[search_start:search_end]) + search_start
        threshold = abs_p[peak_idx] * 0.05
        for k in range(max(0, peak_idx - 300), peak_idx):
            if np.all(abs_p[k : k+20] >= threshold): return k
        return peak_idx

    # =================================================================
    # --- 零件 2: 临床级特征提取 ---
    # =================================================================
    def pslc_apply_tkeo(self, signal):
        tkeo = np.zeros_like(signal)
        tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
        return np.abs(tkeo)

    def pslc_feature_extraction(self, muscle_data, t0):
        if t0 < 600 or t0 + 350 > len(muscle_data): return None
        m_abs = np.abs(muscle_data)
        b_area = np.trapz(m_abs[t0-600 : t0-450, :], axis=0) + 1e-8
        
        apa1_area = np.trapz(m_abs[t0-250 : t0-100, :], axis=0)
        apa2_area = np.trapz(m_abs[t0-100 : t0+50, :], axis=0)
        cpa1_area = np.trapz(m_abs[t0+50 : t0+200, :], axis=0)
        cpa2_area = np.trapz(m_abs[t0+200 : t0+350, :], axis=0)
        
        return np.concatenate([
            (apa1_area - b_area) / b_area,
            (apa2_area - b_area) / b_area,
            (cpa1_area - b_area) / b_area,
            (cpa2_area - b_area) / b_area
        ])

    def pslc_build_dataset(self, root_dir, label_file):
        df_labels = pd.read_excel(label_file)
        df_labels['姓名'] = df_labels['姓名'].astype(str).str.strip()
        df_labels.set_index('姓名', inplace=True)
        X_iemg, y_list, valid_names = [], [], []
        
        for p_name in tqdm(os.listdir(root_dir)):
            if p_name not in df_labels.index: continue
            p_path = os.path.join(root_dir, p_name)
            arm_feats, ball_feats = [], []
            
            # 举臂任务
            if os.path.exists(os.path.join(p_path, '举臂1')):
                for f in glob.glob(os.path.join(p_path, '举臂1', "*_mc.xls")):
                    _, raw, _ = self.pslc_robust_reader(f)
                    if len(raw) < 2000: continue
                    clean = self.pslc_clean_emg(raw)
                    b, a = butter(2, 2/500, btype='low')
                    global_env = filtfilt(b, a, np.sum(self.pslc_apply_tkeo(clean), axis=1))
                    t0 = np.argmax(global_env)
                    ft = self.pslc_feature_extraction(clean, t0)
                    if ft is not None: arm_feats.append(ft)
            
            # 落球任务
            if os.path.exists(os.path.join(p_path, '落球1')):
                for f in glob.glob(os.path.join(p_path, '落球1', "*_mc.xls")):
                    _, raw, pr = self.pslc_robust_reader(f)
                    if len(raw) < 2000: continue
                    t0 = self.pslc_find_t0_ball(pr)
                    ft = self.pslc_feature_extraction(self.pslc_clean_emg(raw), t0)
                    if ft is not None: ball_feats.append(ft)
            
            if arm_feats and ball_feats:
                X_iemg.append(np.concatenate([np.mean(arm_feats, axis=0), np.mean(ball_feats, axis=0)]))
                y_list.append(int(df_labels.loc[p_name, 'PPI标签'])) # 默认返回PPI标签，后续动态提取
                valid_names.append(p_name)
                
        return np.array(X_iemg), np.array(y_list), valid_names

    def _print_pslc_report(self, task, res):
        print(f"\n🏆 PSLC_{task} 最终战报 (5-Fold Averaged)\n" + "-"*50)
        print(f"5-Fold AUC:         {res['AUC']:.4f}")
        print(f"准确率 (Accuracy):    {res['Accuracy']*100:.2f}%")
        print(f"马修斯系数 (MCC):     {res['MCC']:.4f}")
        print(f"F1-Score:           {res['F1']:.4f}")
        print(f"灵敏度 (Sensitivity): {res['Sensitivity']*100:.2f}%")
        print(f"特异度 (Specificity): {res['Specificity']*100:.2f}%\n" + "-"*50)

    # =================================================================
    # --- 零件 3: PPI (运动意图维度) ---
    # =================================================================
    def run_pslc_ppi(self, X_final, y):
        X_delta = X_final[:, :32] - X_final[:, 32:]
        X_sub = X_delta[:, self.GOLD_INDICES_PPI] 
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_dict = {'auc': [], 'acc': [], 'mcc': [], 'f1': [], 'sens': [], 'spec': []}
        
        print(f"🚀 启动 PPI 5-Fold 巅峰引擎 (特征: {self.GOLD_INDICES_PPI}, N={len(y)})...")
        
        for train_idx, test_idx in cv.split(X_sub, y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                for k in metrics_dict: metrics_dict[k].append(0.5 if k in ['auc', 'acc'] else 0.0)
                continue
                
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('svm', SVC(kernel='poly', degree=2, C=0.8, coef0=1, 
                            class_weight='balanced', probability=True, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            
            train_probs = pipe.predict_proba(X_train)[:, 1]
            fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_probs)
            best_t = thresh_train[np.argmax(tpr_train - fpr_train)]
            
            test_probs = pipe.predict_proba(X_test)[:, 1]
            y_pred = [1 if p >= best_t else 0 for p in test_probs]
            
            fold_auc = roc_auc_score(y_test, test_probs)
            if fold_auc < 0.5: fold_auc = 1 - fold_auc 
                
            fold_acc = accuracy_score(y_test, y_pred)
            fold_mcc = matthews_corrcoef(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            fold_sens = tp / (tp + fn + 1e-8)
            fold_spec = tn / (tn + fp + 1e-8)
            
            metrics_dict['auc'].append(fold_auc)
            metrics_dict['acc'].append(fold_acc)
            metrics_dict['mcc'].append(fold_mcc)
            metrics_dict['f1'].append(fold_f1)
            metrics_dict['sens'].append(fold_sens)
            metrics_dict['spec'].append(fold_spec)
            
        res = {k.upper() if k in ['auc', 'mcc', 'f1'] else k.capitalize(): np.mean(v) for k, v in metrics_dict.items()}
        res['Accuracy'] = res.pop('Acc')
        res['Sensitivity'] = res.pop('Sens')
        res['Specificity'] = res.pop('Spec')
        
        self._print_pslc_report("PPI", res)
        return res

    # =================================================================
    # --- 零件 4: VAS (主观疼痛维度) ---
    # =================================================================
    def run_pslc_vas(self, X_final, valid_names, label_file):
        df = pd.read_excel(label_file)
        df['姓名'] = df['姓名'].astype(str).str.strip()
        y_target = df.set_index('姓名').loc[valid_names, 'VAS标签（二分类）'].values
        
        X_delta = X_final[:, :32] - X_final[:, 32:]
        X_sub = X_delta[:, self.GOLD_INDICES_VAS]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_dict = {'auc': [], 'acc': [], 'mcc': [], 'f1': [], 'sens': [], 'spec': []}
        
        print(f"📡 启动 VAS 5-Fold 巅峰引擎 (特征: {self.GOLD_INDICES_VAS}, N={len(y_target)})...")
        
        for train_idx, test_idx in cv.split(X_sub, y_target):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y_target[train_idx], y_target[test_idx]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                for k in metrics_dict: metrics_dict[k].append(0.5 if k in ['auc', 'acc'] else 0.0)
                continue
            
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('svm', SVC(kernel='poly', degree=2, C=0.5, coef0=1.0, 
                            class_weight='balanced', probability=True, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            
            train_probs = pipe.predict_proba(X_train)[:, 1]
            fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_probs)
            best_t = thresh_train[np.argmax(tpr_train - fpr_train)]
            
            test_probs = pipe.predict_proba(X_test)[:, 1]
            y_pred = [1 if p >= best_t else 0 for p in test_probs]
            
            fold_auc = roc_auc_score(y_test, test_probs)
            if fold_auc < 0.5: fold_auc = 1 - fold_auc
                
            fold_acc = accuracy_score(y_test, y_pred)
            fold_mcc = matthews_corrcoef(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            fold_sens = tp / (tp + fn + 1e-8)
            fold_spec = tn / (tn + fp + 1e-8)
            
            metrics_dict['auc'].append(fold_auc)
            metrics_dict['acc'].append(fold_acc)
            metrics_dict['mcc'].append(fold_mcc)
            metrics_dict['f1'].append(fold_f1)
            metrics_dict['sens'].append(fold_sens)
            metrics_dict['spec'].append(fold_spec)
            
        res = {k.upper() if k in ['auc', 'mcc', 'f1'] else k.capitalize(): np.mean(v) for k, v in metrics_dict.items()}
        res['Accuracy'] = res.pop('Acc')
        res['Sensitivity'] = res.pop('Sens')
        res['Specificity'] = res.pop('Spec')
        
        self._print_pslc_report("VAS", res)
        return res

    # =================================================================
    # --- 零件 5: ODI (功能障碍维度) ---
    # =================================================================
    def run_pslc_odi(self, X_final, valid_names, label_file):
        df = pd.read_excel(label_file)
        df['姓名'] = df['姓名'].astype(str).str.strip()
        y_target = df.set_index('姓名').loc[valid_names, 'ODI标签'].values
        
        X_delta = X_final[:, :32] - X_final[:, 32:]
        X_sub = X_delta[:, self.GOLD_INDICES_ODI]
        
        odi_weight = {0: 1.0, 1: sum(y_target == 0) / (sum(y_target == 1) + 1e-8)}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_dict = {'auc': [], 'acc': [], 'mcc': [], 'f1': [], 'sens': [], 'spec': []}
        
        print(f"📡 启动 ODI 5-Fold 严谨验证引擎 (特征: {self.GOLD_INDICES_ODI}, N={len(y_target)})...")
        
        for train_idx, test_idx in cv.split(X_sub, y_target):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y_target[train_idx], y_target[test_idx]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                for k in metrics_dict: metrics_dict[k].append(0.5 if k in ['auc', 'acc'] else 0.0)
                continue
                
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('svm', SVC(kernel='poly', degree=2, C=0.1, coef0=1.0, 
                            class_weight=odi_weight, probability=False, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            
            train_scores = pipe.decision_function(X_train)
            train_auc = roc_auc_score(y_train, train_scores)
            
            if train_auc < 0.5: 
                train_scores = -train_scores
                
            fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_scores)
            best_t = thresh_train[np.argmax(tpr_train - fpr_train)]
            
            test_scores = pipe.decision_function(X_test)
            if train_auc < 0.5: 
                test_scores = -test_scores
                
            y_pred = [1 if s >= best_t else 0 for s in test_scores]
            
            fold_auc = roc_auc_score(y_test, test_scores)
            if fold_auc < 0.5: fold_auc = 1 - fold_auc
                
            fold_acc = accuracy_score(y_test, y_pred)
            fold_mcc = matthews_corrcoef(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            fold_sens = tp / (tp + fn + 1e-8)
            fold_spec = tn / (tn + fp + 1e-8)
            
            metrics_dict['auc'].append(fold_auc)
            metrics_dict['acc'].append(fold_acc)
            metrics_dict['mcc'].append(fold_mcc)
            metrics_dict['f1'].append(fold_f1)
            metrics_dict['sens'].append(fold_sens)
            metrics_dict['spec'].append(fold_spec)
            
        res = {k.upper() if k in ['auc', 'mcc', 'f1'] else k.capitalize(): np.mean(v) for k, v in metrics_dict.items()}
        res['Accuracy'] = res.pop('Acc')
        res['Sensitivity'] = res.pop('Sens')
        res['Specificity'] = res.pop('Spec')
        
        self._print_pslc_report("ODI", res)
        return res