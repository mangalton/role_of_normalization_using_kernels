
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import time
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Dataset loaded successfully!")
print(f"Training set shape: {X_train_full.shape}")
print(f"Training labels shape: {y_train_full.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"\nNumber of classes: {len(class_names)}")
print(f"Classes: {class_names}")


y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i])
    ax.set_title(f"Class: {class_names[y_train_full[i]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()


unique, counts = np.unique(y_train_full, return_counts=True)
plt.figure(figsize=(10, 5))
plt.bar(class_names, counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution in Training Set')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


n_train_samples = 50000
n_test_samples = 10000

def stratified_sample(X, y, n_samples):
    """Sample evenly from each class"""
    n_classes = len(np.unique(y))
    samples_per_class = n_samples // n_classes

    indices = []
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(selected)

    return X[indices], y[indices]

X_train, y_train = stratified_sample(X_train_full, y_train_full, n_train_samples)
X_test_subset, y_test_subset = stratified_sample(X_test, y_test, n_test_samples)

print(f"Training subset shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test subset shape: {X_test_subset.shape}")
print(f"Test labels shape: {y_test_subset.shape}")

print("\nClass distribution in training subset:")
for i, name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"{name}: {count}")

X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32')
X_test_flat = X_test_subset.reshape(X_test_subset.shape[0], -1).astype('float32')

print(f"Flattened training shape: {X_train_flat.shape}")
print(f"Flattened test shape: {X_test_flat.shape}")



X_train_processed = X_train_flat
X_test_processed = X_test_flat

print(f"Training data shape: {X_train_processed.shape}")
print(f"Test data shape: {X_test_processed.shape}")


def apply_normalization(X_train, X_test, method='standard'):
    """
    Apply different normalization methods
    
    Parameters:
    - method: 'none', 'standard', 'minmax', 'robust', 'l2'
    
    Returns:
    - X_train_norm, X_test_norm, scaler object (or None)
    """
    scaler = None  
    
    if method == 'none':
        return X_train.copy(), X_test.copy(), scaler

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
        scaler = None 

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm, scaler


normalization_methods = ['none', 'standard', 'minmax', 'robust', 'l2']

print("Testing normalization methods on a sample...")
for method in normalization_methods:
    X_tr, X_te , _ = apply_normalization(X_train_processed, X_test_processed, method)
    print(f"\n{method.upper()} normalization:")
    print(f"  Train - Mean: {X_tr.mean():.4f}, Std: {X_tr.std():.4f}, Min: {X_tr.min():.4f}, Max: {X_tr.max():.4f}")
    print(f"  Test  - Mean: {X_te.mean():.4f}, Std: {X_te.std():.4f}, Min: {X_te.min():.4f}, Max: {X_te.max():.4f}")

kernels = ['linear', 'rbf', 'poly']
normalization_methods = ['none', 'standard', 'minmax', 'robust', 'l2']

results = []

print("Training SVM models with different kernels and normalizations...")
print("=" * 80)

for norm_method in normalization_methods:
    print(f"\nNormalization: {norm_method.upper()}")
    print("-" * 80)

    X_train_norm, X_test_norm , _ = apply_normalization(X_train_processed, X_test_processed, norm_method)

    for kernel in kernels:
        print(f"  Kernel: {kernel}...", end=' ')

        start_time = time.time()

        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42, max_iter=1000)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42, max_iter=1000)

        svm.fit(X_train_norm, y_train)

        # Predictions
        y_pred_train = svm.predict(X_train_norm)
        y_pred_test = svm.predict(X_test_norm)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test_subset, y_pred_test)

        training_time = time.time() - start_time

        results.append({
            'normalization': norm_method,
            'kernel': kernel,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'training_time': training_time
        })

        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {training_time:.2f}s")

print("\n" + "=" * 80)
print("Training complete!")

import pandas as pd

results_df = pd.DataFrame(results)

print("RESULTS SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))
print()

best_idx = results_df['test_accuracy'].idxmax()
best_config = results_df.iloc[best_idx]
print(f"\nBest Configuration:")
print(f"  Normalization: {best_config['normalization']}")
print(f"  Kernel: {best_config['kernel']}")
print(f"  Test Accuracy: {best_config['test_accuracy']:.4f}")
print(f"  Training Time: {best_config['training_time']:.2f}s")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

ax1 = axes[0, 0]
pivot_acc = results_df.pivot(index='normalization', columns='kernel', values='test_accuracy')
pivot_acc.plot(kind='bar', ax=ax1)
ax1.set_title('Test Accuracy by Normalization Method and Kernel', fontsize=12, fontweight='bold')
ax1.set_xlabel('Normalization Method')
ax1.set_ylabel('Test Accuracy')
ax1.legend(title='Kernel')
ax1.grid(True, alpha=0.3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

ax2 = axes[0, 1]
pivot_time = results_df.pivot(index='normalization', columns='kernel', values='training_time')
pivot_time.plot(kind='bar', ax=ax2)
ax2.set_title('Training Time by Normalization Method and Kernel', fontsize=12, fontweight='bold')
ax2.set_xlabel('Normalization Method')
ax2.set_ylabel('Training Time (seconds)')
ax2.legend(title='Kernel')
ax2.grid(True, alpha=0.3)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

ax3 = axes[1, 0]
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax3, cbar_kws={'label': 'Accuracy'})
ax3.set_title('Test Accuracy Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Kernel')
ax3.set_ylabel('Normalization Method')

ax4 = axes[1, 1]
for kernel in kernels:
    kernel_data = results_df[results_df['kernel'] == kernel]
    ax4.plot(range(len(kernel_data)), kernel_data['test_accuracy'],
             marker='o', label=kernel, linewidth=2)
ax4.set_xticks(range(len(normalization_methods)))
ax4.set_xticklabels(normalization_methods, rotation=45, ha='right')
ax4.set_title('Test Accuracy Across Normalization Methods', fontsize=12, fontweight='bold')
ax4.set_xlabel('Normalization Method')
ax4.set_ylabel('Test Accuracy')
ax4.legend(title='Kernel')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Use the best configuration found
best_norm = best_config['normalization']
best_kernel = best_config['kernel']

print(f"Training best model: {best_kernel} kernel with {best_norm} normalization")
print("=" * 80)

X_train_best, X_test_best, _ = apply_normalization(X_train_processed, X_test_processed, best_norm)

if best_kernel == 'poly':
    best_svm = SVC(kernel=best_kernel, degree=3, C=1.0, random_state=42, max_iter=1000)
else:
    best_svm = SVC(kernel=best_kernel, C=1.0, random_state=42, max_iter=1000)

best_svm.fit(X_train_best, y_train)

y_pred_train_best = best_svm.predict(X_train_best)
y_pred_test_best = best_svm.predict(X_test_best)

train_acc_best = accuracy_score(y_train, y_pred_train_best)
test_acc_best = accuracy_score(y_test_subset, y_pred_test_best)

print(f"\nBest Model Performance:")
print(f"  Training Accuracy: {train_acc_best:.4f}")
print(f"  Test Accuracy: {test_acc_best:.4f}")
print(f"  Number of Support Vectors: {len(best_svm.support_)}")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test_subset, y_pred_test_best, target_names=class_names))

cm = confusion_matrix(y_test_subset, y_pred_test_best)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_kernel.upper()} Kernel with {best_norm.upper()} Normalization',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(class_accuracies):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

import joblib
best_norm = best_config['normalization']
best_kernel = best_config['kernel']

print(f"Training best model: {best_kernel} kernel with {best_norm} normalization")
print("=" * 80)

X_train_best, X_test_best, best_scaler = apply_normalization(
    X_train_processed, X_test_processed, best_norm
)


print(f"Scaler for best model: {best_scaler}")

if best_kernel == 'poly':
    best_svm = SVC(kernel=best_kernel, degree=3, C=1.0, random_state=42, max_iter=1000)
else:
    best_svm = SVC(kernel=best_kernel, C=1.0, random_state=42, max_iter=1000)

best_svm.fit(X_train_best, y_train)


model_filename = 'best_svm_model.joblib'
joblib.dump(best_svm, model_filename)
print(f"\nSuccessfully saved model to: {model_filename}")


if best_scaler is not None:
    scaler_filename = 'best_scaler.joblib'
    joblib.dump(best_scaler, scaler_filename)
    print(f"Successfully saved scaler to: {scaler_filename}")

print("Hyperparameter Tuning")
print("=" * 80)

C_values = [0.1, 1.0, 10.0]

if best_kernel == 'rbf':
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
    hyperparam_results = []

    for C in C_values:
        for gamma in gamma_values:
            print(f"Testing C={C}, gamma={gamma}...", end=' ')

            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42, max_iter=1000)
           
            svm.fit(X_train_best, y_train)

            y_pred = svm.predict(X_test_best)
            acc = accuracy_score(y_test_subset, y_pred)

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
    sns.heatmap(pivot_hp, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Hyperparameter Search - RBF Kernel', fontsize=12, fontweight='bold')
    plt.xlabel('C (Regularization)')
    plt.ylabel('Gamma')
    plt.tight_layout()
    plt.show()

elif best_kernel == 'poly':
    degree_values = [2, 3, 4]
    hyperparam_results = []

    for C in C_values:
        for degree in degree_values:
            print(f"Testing C={C}, degree={degree}...", end=' ')

            svm = SVC(kernel='poly', C=C, degree=degree, random_state=42, max_iter=1000)
            svm.fit(X_train_best, y_train)

            y_pred = svm.predict(X_test_best)
            acc = accuracy_score(y_test_subset, y_pred)

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

    for C in C_values:
        print(f"Testing C={C}...", end=' ')

        svm = SVC(kernel='linear', C=C, random_state=42, max_iter=1000)
        svm.fit(X_train_best, y_train)

        y_pred = svm.predict(X_test_best)
        acc = accuracy_score(y_test_subset, y_pred)

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

print("Impact of Data Normalization on Kernel Methods")
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
        width, label='Train Accuracy', alpha=0.8)
ax1.bar([i + width/2 for i in x], norm_comparison['test_accuracy'],
        width, label='Test Accuracy', alpha=0.8)
ax1.set_xlabel('Normalization Method')
ax1.set_ylabel('Accuracy')
ax1.set_title(f'Train vs Test Accuracy - {best_kernel.upper()} Kernel', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(norm_comparison['normalization'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
overfit_gap = norm_comparison['train_accuracy'] - norm_comparison['test_accuracy']
colors = ['red' if gap > 0.1 else 'green' for gap in overfit_gap]
ax2.bar(range(len(norm_comparison)), overfit_gap, color=colors, alpha=0.7)
ax2.set_xlabel('Normalization Method')
ax2.set_ylabel('Train-Test Gap')
ax2.set_title('Overfitting Analysis', fontweight='bold')
ax2.set_xticks(range(len(norm_comparison)))
ax2.set_xticklabels(norm_comparison['normalization'], rotation=45, ha='right')
ax2.axhline(y=0.1, color='r', linestyle='--', label='High overfit threshold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes[2]
efficiency = norm_comparison['test_accuracy'] / norm_comparison['training_time']
ax3.bar(range(len(norm_comparison)), efficiency, color='purple', alpha=0.7)
ax3.set_xlabel('Normalization Method')
ax3.set_ylabel('Accuracy / Training Time')
ax3.set_title('Training Efficiency', fontweight='bold')
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


print("Analyzing Misclassified Samples")
print("=" * 80)

misclassified_idx = np.where(y_pred_test_best != y_test_subset)[0]
print(f"Number of misclassified samples: {len(misclassified_idx)} out of {len(y_test_subset)}")
print(f"Misclassification rate: {len(misclassified_idx)/len(y_test_subset)*100:.2f}%")

n_show = min(12, len(misclassified_idx))
fig, axes = plt.subplots(3, 4, figsize=(15, 12))

for i, ax in enumerate(axes.flat):
    if i < n_show:
        idx = misclassified_idx[i]
        original_idx = np.where((X_test_subset == X_test_subset[idx]).all(axis=(1,2,3)))[0][0]

        ax.imshow(X_test_subset[idx])
        true_label = class_names[y_test_subset[idx]]
        pred_label = class_names[y_pred_test_best[idx]]
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')

plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold', y=1.00)
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
                'count': cm[i, j]
            })

confusion_df = pd.DataFrame(confusion_pairs)
confusion_df = confusion_df.sort_values('count', ascending=False)
print(confusion_df.head(10).to_string(index=False))


# Create a final comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax1 = axes[0]
avg_by_norm = results_df.groupby('normalization')['test_accuracy'].mean().sort_values(ascending=False)
colors_norm = plt.cm.viridis(np.linspace(0, 1, len(avg_by_norm)))
ax1.barh(range(len(avg_by_norm)), avg_by_norm.values, color=colors_norm)
ax1.set_yticks(range(len(avg_by_norm)))
ax1.set_yticklabels(avg_by_norm.index)
ax1.set_xlabel('Average Test Accuracy')
ax1.set_title('Normalization Methods Ranking\n(Average across all kernels)', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(avg_by_norm.values):
    ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

ax2 = axes[1]
avg_by_kernel = results_df.groupby('kernel')['test_accuracy'].mean().sort_values(ascending=False)
colors_kernel = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax2.barh(range(len(avg_by_kernel)), avg_by_kernel.values, color=colors_kernel)
ax2.set_yticks(range(len(avg_by_kernel)))
ax2.set_yticklabels(avg_by_kernel.index)
ax2.set_xlabel('Average Test Accuracy')
ax2.set_title('Kernel Methods Ranking\n(Average across all normalizations)', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(avg_by_kernel.values):
    ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

