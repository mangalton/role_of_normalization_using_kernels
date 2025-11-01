
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from tensorflow import keras
import time
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 80)

# ============================================================================
# 1: Load Fashion-MNIST Dataset
# ============================================================================
print("\n" + "=" * 80)
print("LOADING FASHION-MNIST DATASET")
print("=" * 80)

(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset loaded successfully!")
print(f"Training set shape: {X_train_full.shape}")
print(f"Training labels shape: {y_train_full.shape}")
print(f"Test set shape: {X_test_full.shape}")
print(f"Test labels shape: {y_test_full.shape}")
print(f"\nNumber of classes: {len(class_names)}")
print(f"Classes: {class_names}")
print(f"Image size: 28x28 pixels (grayscale)")
print(f"Total training samples: {len(y_train_full)}")
print(f"Total test samples: {len(y_test_full)}")

y_train_full = y_train_full.flatten()
y_test_full = y_test_full.flatten()

# ============================================================================
#  2: Visualize Dataset
# ============================================================================
print("\n" + "=" * 80)
print("DATASET VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i], cmap='gray')
    ax.set_title(f"Class: {class_names[y_train_full[i]]}", fontsize=10)
    ax.axis('off')
plt.suptitle('Fashion-MNIST Sample Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

unique, counts = np.unique(y_train_full, return_counts=True)
plt.figure(figsize=(12, 6))
plt.bar(class_names, counts, color='steelblue', alpha=0.7)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\nClass Distribution:")
for i, name in enumerate(class_names):
    print(f"  {name:15s}: {counts[i]} samples")

# ============================================================================
#  3: Use Full Dataset (No Sampling)
# ============================================================================
print("\n" + "=" * 80)
print("USING FULL DATASET (NO SAMPLING)")
print("=" * 80)

X_train = X_train_full
y_train = y_train_full
X_test = X_test_full
y_test = y_test_full

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Image shape: {X_train[0].shape}")

print("\nClass distribution in training set:")
for i, name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"  {name:15s}: {count}")

# ============================================================================
#  4: Preprocessing (Flatten Only - No PCA)
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING: FLATTENING IMAGES (NO DIMENSIONALITY REDUCTION)")
print("=" * 80)

X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32')
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32')

print(f"Original image shape: {X_train[0].shape}")
print(f"Flattened training shape: {X_train_flat.shape}")
print(f"Flattened test shape: {X_test_flat.shape}")
print(f"Number of features per sample: {X_train_flat.shape[1]}")

X_train_processed = X_train_flat
X_test_processed = X_test_flat

print(f"\nProcessed data statistics:")
print(f"  Training data shape: {X_train_processed.shape}")
print(f"  Test data shape: {X_test_processed.shape}")
print(f"  Min value: {X_train_processed.min():.2f}")
print(f"  Max value: {X_train_processed.max():.2f}")
print(f"  Mean value: {X_train_processed.mean():.2f}")
print(f"  Std dev: {X_train_processed.std():.2f}")

# ============================================================================
#  5: Normalization Methods
# ============================================================================
print("\n" + "=" * 80)
print("NORMALIZATION METHODS")
print("=" * 80)

def apply_normalization(X_train, X_test, method='standard'):
    """
    Apply different normalization methods

    Parameters:
    - method: 'none', 'standard', 'minmax', 'robust', 'l2'
    """
    if method == 'none':
        return X_train.copy(), X_test.copy()

    elif method == 'standard':
        # Standardization (zero mean, unit variance)
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

    elif method == 'minmax':
        # Min-Max scaling (scale to [0, 1])
        scaler = MinMaxScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

    elif method == 'robust':
        # Robust scaling (uses median and IQR, robust to outliers)
        scaler = RobustScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

    elif method == 'l2':
        # L2 normalization (unit norm)
        X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm

normalization_methods = ['none', 'standard', 'minmax', 'robust', 'l2']

print("Testing normalization methods on the dataset...")
print("-" * 80)
for method in normalization_methods:
    X_tr, X_te = apply_normalization(X_train_processed, X_test_processed, method)
    print(f"\n{method.upper()} normalization:")
    print(f"  Train - Mean: {X_tr.mean():.4f}, Std: {X_tr.std():.4f}, Min: {X_tr.min():.4f}, Max: {X_tr.max():.4f}")
    print(f"  Test  - Mean: {X_te.mean():.4f}, Std: {X_te.std():.4f}, Min: {X_te.min():.4f}, Max: {X_te.max():.4f}")

# ============================================================================
#  6: Train SVM with Different Kernels and Normalizations
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SVM MODELS")
print("=" * 80)

kernels = ['linear', 'rbf', 'poly']

results = []

print("\nTraining SVM models with different kernels and normalizations...")
print("This may take some time with the full dataset...")
print("=" * 80)

for norm_method in normalization_methods:
    print(f"\nNormalization: {norm_method.upper()}")
    print("-" * 80)

    X_train_norm, X_test_norm = apply_normalization(X_train_processed, X_test_processed, norm_method)

    for kernel in kernels:
        print(f"  Kernel: {kernel:8s}...", end=' ', flush=True)

        start_time = time.time()

        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42, max_iter=1000)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42, max_iter=1000)

        svm.fit(X_train_norm, y_train)

        y_pred_train = svm.predict(X_train_norm)
        y_pred_test = svm.predict(X_test_norm)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        training_time = time.time() - start_time

        results.append({
            'normalization': norm_method,
            'kernel': kernel,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time,
            'n_support_vectors': len(svm.support_)
        })

        print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}, Time: {training_time:.2f}s, SVs: {len(svm.support_)}")

print("\n" + "=" * 80)
print("Training complete!")
print("=" * 80)

# ============================================================================
#  7: Results Summary and Visualization
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_idx = results_df['test_accuracy'].idxmax()
best_config = results_df.iloc[best_idx]

print(f"\n{'='*80}")
print("BEST CONFIGURATION:")
print(f"{'='*80}")
print(f"  Normalization: {best_config['normalization']}")
print(f"  Kernel: {best_config['kernel']}")
print(f"  Test Accuracy: {best_config['test_accuracy']:.4f} ({best_config['test_accuracy']*100:.2f}%)")
print(f"  Train Accuracy: {best_config['train_accuracy']:.4f}")
print(f"  Training Time: {best_config['training_time']:.2f}s")
print(f"  Support Vectors: {best_config['n_support_vectors']}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

ax1 = axes[0, 0]
pivot_acc = results_df.pivot(index='normalization', columns='kernel', values='test_accuracy')
pivot_acc.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Test Accuracy by Normalization Method and Kernel', fontsize=12, fontweight='bold')
ax1.set_xlabel('Normalization Method')
ax1.set_ylabel('Test Accuracy')
ax1.legend(title='Kernel')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = axes[0, 1]
pivot_time = results_df.pivot(index='normalization', columns='kernel', values='training_time')
pivot_time.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('Training Time by Normalization Method and Kernel', fontsize=12, fontweight='bold')
ax2.set_xlabel('Normalization Method')
ax2.set_ylabel('Training Time (seconds)')
ax2.legend(title='Kernel')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

ax3 = axes[0, 2]
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax3,
            cbar_kws={'label': 'Accuracy'}, vmin=0.7, vmax=0.95)
ax3.set_title('Test Accuracy Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Kernel')
ax3.set_ylabel('Normalization Method')

ax4 = axes[1, 0]
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    x_pos = np.arange(len(kernel_data))
    ax4.plot(x_pos, kernel_data['train_accuracy'], marker='s', label=f'{kernel} (train)', linestyle='--', alpha=0.7)
    ax4.plot(x_pos, kernel_data['test_accuracy'], marker='o', label=f'{kernel} (test)', linewidth=2)
ax4.set_xticks(range(len(normalization_methods)))
ax4.set_xticklabels(normalization_methods, rotation=45, ha='right')
ax4.set_title('Train vs Test Accuracy Across Normalizations', fontsize=12, fontweight='bold')
ax4.set_xlabel('Normalization Method')
ax4.set_ylabel('Accuracy')
ax4.legend(fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)

ax5 = axes[1, 1]
pivot_svs = results_df.pivot(index='normalization', columns='kernel', values='n_support_vectors')
pivot_svs.plot(kind='bar', ax=ax5, width=0.8)
ax5.set_title('Number of Support Vectors', fontsize=12, fontweight='bold')
ax5.set_xlabel('Normalization Method')
ax5.set_ylabel('Number of Support Vectors')
ax5.legend(title='Kernel')
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

ax6 = axes[1, 2]
results_df['efficiency'] = results_df['test_accuracy'] / results_df['training_time']
pivot_eff = results_df.pivot(index='normalization', columns='kernel', values='efficiency')
pivot_eff.plot(kind='bar', ax=ax6, width=0.8)
ax6.set_title('Training Efficiency (Accuracy/Time)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Normalization Method')
ax6.set_ylabel('Efficiency Score')
ax6.legend(title='Kernel')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# ============================================================================
#  8: Train and Analyze Best Model in Detail
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED ANALYSIS OF BEST MODEL")
print("=" * 80)

best_norm = best_config['normalization']
best_kernel = best_config['kernel']

print(f"\nTraining best model: {best_kernel} kernel with {best_norm} normalization")
print("-" * 80)

X_train_best, X_test_best = apply_normalization(X_train_processed, X_test_processed, best_norm)

if best_kernel == 'poly':
    best_svm = SVC(kernel=best_kernel, degree=3, C=1.0, random_state=42, max_iter=1000)
else:
    best_svm = SVC(kernel=best_kernel, C=1.0, random_state=42, max_iter=1000)

best_svm.fit(X_train_best, y_train)

# Predictions
y_pred_train_best = best_svm.predict(X_train_best)
y_pred_test_best = best_svm.predict(X_test_best)

train_acc_best = accuracy_score(y_train, y_pred_train_best)
test_acc_best = accuracy_score(y_test, y_pred_test_best)

print(f"\nBest Model Performance:")
print(f"  Training Accuracy: {train_acc_best:.4f} ({train_acc_best*100:.2f}%)")
print(f"  Test Accuracy: {test_acc_best:.4f} ({test_acc_best*100:.2f}%)")
print(f"  Number of Support Vectors: {len(best_svm.support_)}")
print(f"  Percentage of SVs: {len(best_svm.support_)/len(y_train)*100:.2f}%")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred_test_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_test_best)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_kernel.upper()} Kernel with {best_norm.upper()} Normalization',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
bars = plt.bar(class_names, class_accuracies, color=colors, alpha=0.7)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
for i, (v, bar) in enumerate(zip(class_accuracies, bars)):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
             ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.show()

print("\nPer-Class Accuracy:")
for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
    print(f"  {name:15s}: {acc:.4f} ({acc*100:.2f}%)")

# ============================================================================
#  9: Hyperparameter Tuning
# ============================================================================
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)

C_values = [0.1, 1.0, 10.0]

if best_kernel == 'rbf':
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
    hyperparam_results = []

    print(f"\nTuning RBF kernel hyperparameters...")
    print("-" * 80)

    for C in C_values:
        for gamma in gamma_values:
            print(f"Testing C={C}, gamma={gamma}...", end=' ', flush=True)

            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42, max_iter=1000)
            svm.fit(X_train_best, y_train)

            y_pred = svm.predict(X_test_best)
            acc = accuracy_score(y_test, y_pred)

            hyperparam_results.append({
                'C': C,
                'gamma': gamma,
                'test_accuracy': acc
            })

            print(f"Accuracy: {acc:.4f}")

    hp_df = pd.DataFrame(hyperparam_results)
    best_hp_idx = hp_df['test_accuracy'].idxmax()
    best_hp = hp_df.iloc[best_hp_idx]

    print("\n" + "=" * 80)
    print(f"Best Hyperparameters:")
    print(f"  C: {best_hp['C']}")
    print(f"  gamma: {best_hp['gamma']}")
    print(f"  Test Accuracy: {best_hp['test_accuracy']:.4f}")

    pivot_hp = hp_df.pivot(index='gamma', columns='C', values='test_accuracy')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_hp, annot=True, fmt='.4f', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Hyperparameter Search - RBF Kernel', fontsize=14, fontweight='bold')
    plt.xlabel('C (Regularization)', fontsize=12)
    plt.ylabel('Gamma', fontsize=12)
    plt.tight_layout()
    plt.show()

elif best_kernel == 'poly':
    degree_values = [2, 3, 4]
    hyperparam_results = []

    print(f"\nTuning Polynomial kernel hyperparameters...")
    print("-" * 80)

    for C in C_values:
        for degree in degree_values:
            print(f"Testing C={C}, degree={degree}...", end=' ', flush=True)

            svm = SVC(kernel='poly', C=C, degree=degree, random_state=42, max_iter=1000)
            svm.fit(X_train_best, y_train)

            y_pred = svm.predict(X_test_best)
            acc = accuracy_score(y_test, y_pred)

            hyperparam_results.append({
                'C': C,
                'degree': degree,
                'test_accuracy': acc
            })

            print(f"Accuracy: {acc:.4f}")

    hp_df = pd.DataFrame(hyperparam_results)
    best_hp_idx = hp_df['test_accuracy'].idxmax()
    best_hp = hp_df.iloc[best_hp_idx]

    print("\n" + "=" * 80)
    print(f"Best Hyperparameters:")
    print(f"  C: {best_hp['C']}")
    print(f"  degree: {best_hp['degree']}")
    print(f"  Test Accuracy: {best_hp['test_accuracy']:.4f}")

else:  
    hyperparam_results = []

    print(f"\nTuning Linear kernel hyperparameters...")
    print("-" * 80)

    for C in C_values:
        print(f"Testing C={C}...", end=' ', flush=True)

        svm = SVC(kernel='linear', C=C, random_state=42, max_iter=1000)
        svm.fit(X_train_best, y_train)

        y_pred = svm.predict(X_test_best)
        acc = accuracy_score(y_test, y_pred)

        hyperparam_results.append({
            'C': C,
            'test_accuracy': acc
        })

        print(f"Accuracy: {acc:.4f}")

    hp_df = pd.DataFrame(hyperparam_results)
    best_hp_idx = hp_df['test_accuracy'].idxmax()
    best_hp = hp_df.iloc[best_hp_idx]

    print("\n" + "=" * 80)
    print(f"Best Hyperparameters:")
    print(f"  C: {best_hp['C']}")
    print(f"  Test Accuracy: {best_hp['test_accuracy']:.4f}")

# ============================================================================
#  10: Impact of Normalization Analysis
# ============================================================================
print("\n" + "=" * 80)
print("IMPACT OF DATA NORMALIZATION ON KERNEL METHODS")
print("=" * 80)

norm_comparison = results_df[results_df['kernel'] == best_kernel].copy()
norm_comparison = norm_comparison.sort_values('test_accuracy', ascending=False)

print(f"\n{best_kernel.upper()} Kernel Performance with Different Normalizations:")
print("-" * 80)
print(norm_comparison[['normalization', 'test_accuracy', 'train_accuracy', 'training_time']].to_string(index=False))

baseline_acc = norm_comparison[norm_comparison['normalization'] == 'none']['test_accuracy'].values[0]
best_norm_acc = norm_comparison['test_accuracy'].max()
improvement = ((best_norm_acc - baseline_acc) / baseline_acc) * 100

print(f"\nImprovement over no normalization: {improvement:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = axes[0]
x = range(len(norm_comparison))
width = 0.35
ax1.bar([i - width/2 for i in x], norm_comparison['train_accuracy'],
        width, label='Train Accuracy', alpha=0.8, color='steelblue')
ax1.bar([i + width/2 for i in x], norm_comparison['test_accuracy'],
        width, label='Test Accuracy', alpha=0.8, color='coral')
ax1.set_xlabel('Normalization Method', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title(f'Train vs Test Accuracy - {best_kernel.upper()} Kernel', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(norm_comparison['normalization'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.7, 1.0])

ax2 = axes[1]
overfit_gap = norm_comparison['train_accuracy'] - norm_comparison['test_accuracy']
colors = ['red' if gap > 0.05 else 'green' for gap in overfit_gap]
ax2.bar(range(len(norm_comparison)), overfit_gap, color=colors, alpha=0.7)
ax2.set_xlabel('Normalization Method', fontsize=12)
ax2.set_ylabel('Train-Test Gap', fontsize=12)
ax2.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(norm_comparison)))
ax2.set_xticklabels(norm_comparison['normalization'], rotation=45, ha='right')
ax2.axhline(y=0.05, color='r', linestyle='--', label='High overfit threshold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes[2]
efficiency = norm_comparison['test_accuracy'] / norm_comparison['training_time']
colors = plt.cm.viridis(np.linspace(0, 1, len(norm_comparison)))
ax3.bar(range(len(norm_comparison)), efficiency, color=colors, alpha=0.7)
ax3.set_xlabel('Normalization Method', fontsize=12)
ax3.set_ylabel('Accuracy / Training Time', fontsize=12)
ax3.set_title('Training Efficiency', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(norm_comparison)))
ax3.set_xticklabels(norm_comparison['normalization'], rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("Statistical Summary of Normalization Impact:")
print("-" * 80)
print(f"Mean test accuracy across normalizations: {norm_comparison['test_accuracy'].mean():.4f}")
print(f"Std dev of test accuracy: {norm_comparison['test_accuracy'].std():.4f}")
print(f"Best normalization: {norm_comparison.iloc[0]['normalization']}")
print(f"Worst normalization: {norm_comparison.iloc[-1]['normalization']}")
print(f"Performance spread: {(norm_comparison['test_accuracy'].max() - norm_comparison['test_accuracy'].min()):.4f}")

# ============================================================================
#  11: Analyze Misclassified Samples
# ============================================================================
print("\n" + "=" * 80)
print("ANALYZING MISCLASSIFIED SAMPLES")
print("=" * 80)

misclassified_idx = np.where(y_pred_test_best != y_test)[0]
print(f"\nNumber of misclassified samples: {len(misclassified_idx)} out of {len(y_test)}")
print(f"Misclassification rate: {len(misclassified_idx)/len(y_test)*100:.2f}%")
print(f"Correct classifications: {len(y_test) - len(misclassified_idx)}")

n_show = min(16, len(misclassified_idx))
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i, ax in enumerate(axes.flat):
    if i < n_show:
        idx = misclassified_idx[i]
        ax.imshow(X_test[idx], cmap='gray')
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred_test_best[idx]]
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')

plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("Most Common Confusion Patterns:")
print("-" * 80)

confusion_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true_class': class_names[i],
                'predicted_class': class_names[j],
                'count': cm[i, j],
                'percentage': (cm[i, j] / cm[i].sum()) * 100
            })

confusion_df = pd.DataFrame(confusion_pairs)
confusion_df = confusion_df.sort_values('count', ascending=False)
print(confusion_df.head(15).to_string(index=False))

top_confusions = confusion_df.head(10)
plt.figure(figsize=(14, 6))
labels = [f"{row['true_class'][:8]}\n→\n{row['predicted_class'][:8]}"
          for _, row in top_confusions.iterrows()]
plt.bar(range(len(top_confusions)), top_confusions['count'], color='crimson', alpha=0.7)
plt.xlabel('Confusion Pair (True → Predicted)', fontsize=12)
plt.ylabel('Number of Misclassifications', fontsize=12)
plt.title('Top 10 Confusion Patterns', fontsize=14, fontweight='bold')
plt.xticks(range(len(top_confusions)), labels, fontsize=9, rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ============================================================================
#  12: Comparison Across All Configurations
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON ACROSS ALL CONFIGURATIONS")
print("=" * 80)

# Overall kernel performance
print("\nAverage Performance by Kernel (across all normalizations):")
print("-" * 80)
kernel_avg = results_df.groupby('kernel').agg({
    'test_accuracy': ['mean', 'std', 'max', 'min'],
    'training_time': ['mean', 'std']
}).round(4)
print(kernel_avg)

print("\n" + "=" * 80)
print("Average Performance by Normalization (across all kernels):")
print("-" * 80)
norm_avg = results_df.groupby('normalization').agg({
    'test_accuracy': ['mean', 'std', 'max', 'min'],
    'training_time': ['mean', 'std']
}).round(4)
print(norm_avg)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
avg_by_norm = results_df.groupby('normalization')['test_accuracy'].mean().sort_values(ascending=False)
colors_norm = plt.cm.viridis(np.linspace(0, 1, len(avg_by_norm)))
bars1 = ax1.barh(range(len(avg_by_norm)), avg_by_norm.values, color=colors_norm, alpha=0.8)
ax1.set_yticks(range(len(avg_by_norm)))
ax1.set_yticklabels(avg_by_norm.index)
ax1.set_xlabel('Average Test Accuracy', fontsize=12)
ax1.set_title('Normalization Methods Ranking\n(Average across all kernels)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, (v, bar) in enumerate(zip(avg_by_norm.values, bars1)):
    ax1.text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=10)

ax2 = axes[0, 1]
avg_by_kernel = results_df.groupby('kernel')['test_accuracy'].mean().sort_values(ascending=False)
colors_kernel = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars2 = ax2.barh(range(len(avg_by_kernel)), avg_by_kernel.values, color=colors_kernel, alpha=0.8)
ax2.set_yticks(range(len(avg_by_kernel)))
ax2.set_yticklabels(avg_by_kernel.index)
ax2.set_xlabel('Average Test Accuracy', fontsize=12)
ax2.set_title('Kernel Methods Ranking\n(Average across all normalizations)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, (v, bar) in enumerate(zip(avg_by_kernel.values, bars2)):
    ax2.text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=10)

ax3 = axes[1, 0]
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    ax3.scatter(kernel_data['training_time'], kernel_data['test_accuracy'],
                s=150, alpha=0.7, label=kernel, marker='o')
ax3.set_xlabel('Training Time (seconds)', fontsize=12)
ax3.set_ylabel('Test Accuracy', fontsize=12)
ax3.set_title('Training Time vs Accuracy Trade-off', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

