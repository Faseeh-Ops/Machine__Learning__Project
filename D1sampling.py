
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def apply_rus(X, y):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)

def plot_distributions(y_original, y_smote, y_rus):
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    sns.countplot(x=y_original, ax=axs[0])
    axs[0].set_title("Original")
    sns.countplot(x=y_smote, ax=axs[1])
    axs[1].set_title("After SMOTE")
    sns.countplot(x=y_rus, ax=axs[2])
    axs[2].set_title("After RUS")
    plt.show()
