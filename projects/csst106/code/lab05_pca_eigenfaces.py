"""Lab 5 - Facial Recognition using PCA (Eigenfaces) + SVM on LFW."""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "img")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close()


# ===========================================================================
# Task 1: Data acquisition and preprocessing
# ===========================================================================
print("Downloading LFW dataset (this may take a moment)...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              data_home=os.path.join(HERE, "data", "sklearn"))

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data              # each row = one flattened h*w face
y = lfw_people.target            # integer identity label
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print(f"n_samples: {n_samples}")
print(f"n_features: {X.shape[1]} ({h}x{w} pixels)")
print(f"n_classes: {n_classes}")
for i, name in enumerate(target_names):
    print(f"  {i}: {name:20s} {np.sum(y == i)} images")

# 80 % train / 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train {X_train.shape}, X_test {X_test.shape}")

# ===========================================================================
# Task 2: Computing the principal components
# ===========================================================================
n_components = 150
print(f"Extracting the top {n_components} eigenfaces from {X_train.shape[0]} faces...")
t0 = time.time()
# Fit ONLY on training data so no information leaks from the test set.
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=42).fit(X_train)
print(f"Done in {time.time() - t0:.3f}s")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"X_train_pca {X_train_pca.shape}, X_test_pca {X_test_pca.shape}")
print(f"variance retained by 150 components: {pca.explained_variance_ratio_.sum():.4f}")
evr = pca.explained_variance_ratio_
print(f"explained variance ratio  PC1={evr[0]:.4f} PC2={evr[1]:.4f} PC3={evr[2]:.4f} (first 3 together {evr[:3].sum():.4f})")
print(f"explained variance ratio  last 3 = {np.round(evr[-3:], 5)}")

# ===========================================================================
# Task 3: Visualising the eigenfaces
# ===========================================================================
eigenfaces = pca.components_.reshape((n_components, h, w))


def plot_gallery(images, titles, h, w, n_row=3, n_col=4, fname=None):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())
    save(fname)


eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w, fname="lab05_eigenfaces.png")

# Mean face + first 3 vs last 3 eigenfaces (for Q2)
plot_gallery(np.concatenate([pca.mean_[None], pca.components_[:3], pca.components_[-4:]]),
             ["mean face", "eigenface 0", "eigenface 1", "eigenface 2",
              "eigenface 146", "eigenface 147", "eigenface 148", "eigenface 149"],
             h, w, n_row=2, n_col=4, fname="lab05_first_vs_last.png")

# ===========================================================================
# Task 4: Face recognition in the reduced space
# ===========================================================================
print("Fitting the classifier to the training set")
t0 = time.time()
clf = SVC(kernel="rbf", class_weight="balanced").fit(X_train_pca, y_train)
print(f"Done in {time.time() - t0:.3f}s")

print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
print(f"accuracy (default SVC) = {accuracy_score(y_test, y_pred):.4f}")

# Improvement: tune C and gamma with 5-fold cross-validation on the TRAINING set only
param_grid = {"C": [1, 5, 10, 50, 100, 1000], "gamma": [0.0005, 0.001, 0.005, 0.01, "scale"]}
t0 = time.time()
grid = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, cv=5, n_jobs=-1).fit(X_train_pca, y_train)
print(f"GridSearch done in {time.time() - t0:.1f}s, best params: {grid.best_params_}, "
      f"cv acc={grid.best_score_:.4f}")
y_pred_tuned = grid.predict(X_test_pca)
print(classification_report(y_test, y_pred_tuned, target_names=target_names, zero_division=0))
print(f"accuracy (tuned SVC) = {accuracy_score(y_test, y_pred_tuned):.4f}")

cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(7.5, 6.2))
plt.imshow(cm, cmap="Blues")
short = [n.split()[-1] for n in target_names]
plt.xticks(range(n_classes), short, rotation=45, ha="right"); plt.yticks(range(n_classes), short)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion matrix (tuned SVC)"); plt.colorbar()
save("lab05_confusion.png")

# ===========================================================================
# Post-lab Q1: Dimensionality vs variance
# ===========================================================================
pca_full = PCA(svd_solver="full").fit(X_train)            # all components, for the full curve
cum = pca_full.explained_variance_ratio_.cumsum()
k90 = int(np.argmax(cum >= 0.90)) + 1
k95 = int(np.argmax(cum >= 0.95)) + 1
k99 = int(np.argmax(cum >= 0.99)) + 1
print(f"components for 90% variance: {k90}, 95%: {k95}, 99%: {k99}; "
      f"100 components keep {cum[99]:.4f}, 150 components keep {cum[149]:.4f}")

plt.figure(figsize=(8.5, 4.8))
plt.plot(np.arange(1, len(cum) + 1), cum, color="navy")
plt.axhline(0.90, color="crimson", ls="--", lw=1)
plt.axvline(k90, color="crimson", ls="--", lw=1)
plt.annotate(f"90 % at k = {k90}", (k90, 0.90), xytext=(k90 + 60, 0.72),
             arrowprops=dict(arrowstyle="->"), color="crimson")
plt.axvline(150, color="green", ls=":", lw=1); plt.text(155, 0.45, f"k=150 -> {cum[149]:.1%}", color="green")
plt.xscale("log"); plt.xlabel("number of components (log scale)"); plt.ylabel("cumulative explained variance")
plt.title("Cumulative explained variance - LFW (70+ faces/person)"); plt.grid(alpha=0.3, which="both")
save("lab05_cumvar.png")

# Reconstructions with an increasing number of components (information loss)
face = X_test[0]
recon_imgs, recon_titles = [face], ["original (1850-D)"]
for k in [10, 50, 150, k90]:
    p = PCA(n_components=k, svd_solver="randomized", random_state=42).fit(X_train)
    rec = p.inverse_transform(p.transform(face[None]))[0]
    recon_imgs.append(rec); recon_titles.append(f"k={k}")
plot_gallery(recon_imgs, recon_titles, h, w, n_row=1, n_col=5, fname="lab05_reconstruction.png")

# Whitening check (Q3): variance of each projected feature
p_nw = PCA(n_components=150, svd_solver="randomized", whiten=False, random_state=42).fit(X_train)
var_nw = p_nw.transform(X_train).var(axis=0)
var_w = X_train_pca.var(axis=0)
print(f"un-whitened PC variances: first={var_nw[0]:.1f}, last={var_nw[-1]:.3f}, ratio={var_nw[0] / var_nw[-1]:.0f}x")
print(f"whitened   PC variances: first={var_w[0]:.3f}, last={var_w[-1]:.3f}")
clf_nw = SVC(kernel="rbf", class_weight="balanced").fit(p_nw.transform(X_train), y_train)
acc_nw = accuracy_score(y_test, clf_nw.predict(p_nw.transform(X_test)))
print(f"accuracy WITHOUT whitening (default SVC) = {acc_nw:.4f}  vs  WITH whitening = "
      f"{accuracy_score(y_test, y_pred):.4f}")

# ===========================================================================
# Post-lab Q4: Failure analysis
# ===========================================================================
wrong = np.where(y_pred_tuned != y_test)[0]
print(f"misclassified test faces: {len(wrong)} / {len(y_test)}")
n_show = min(12, len(wrong))
titles = [f"pred: {target_names[y_pred_tuned[i]].split()[-1]}\ntrue: {target_names[y_test[i]].split()[-1]}"
          for i in wrong[:n_show]]
plot_gallery(X_test[wrong[:n_show]], titles, h, w, n_row=int(np.ceil(n_show / 6)), n_col=6,
             fname="lab05_failures.png")

# Quantitative hints for the hypotheses: brightness, contrast, left/right lighting asymmetry,
# and reconstruction error (how "face-like" the image is for this PCA model)
def describe(idx):
    f = X_test[idx].reshape(h, w)
    asym = f[:, : w // 2].mean() - f[:, w // 2:].mean()
    rec = pca.inverse_transform(pca.transform(X_test[idx][None]))[0]
    return f.mean(), f.std(), abs(asym), np.linalg.norm(X_test[idx] - rec)

ok = np.where(y_pred_tuned == y_test)[0]
d_wrong = np.array([describe(i) for i in wrong])
d_ok = np.array([describe(i) for i in ok])
for k, name in enumerate(["mean brightness", "contrast (std)", "|left-right lighting asym|", "PCA recon. error"]):
    print(f"{name:28s} wrong={d_wrong[:, k].mean():7.2f}   correct={d_ok[:, k].mean():7.2f}")
for i in wrong[:n_show]:
    print(f"  test idx {i:3d}: true={target_names[y_test[i]]:18s} pred={target_names[y_pred_tuned[i]]}")
# Which classes are confused most? (class imbalance)
counts = np.bincount(y_train, minlength=n_classes)
for i, n in enumerate(target_names):
    print(f"  train images {n:20s} {counts[i]}")
