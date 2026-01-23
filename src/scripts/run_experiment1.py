import random
import torch as th
from torchvision import transforms
import copy
import csv
import argparse
import os

# Simple Pytorch FFNN with hyperparameters controlling number of layers and hidden units per layer
class SimpleFFNN(th.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFFNN, self).__init__()
        self.fc1 = th.nn.Linear(input_size, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.sigmoid(self.fc2(x))
        return x
# Function that returns empirical risk on data X,Y passed into it for a given model
def get_empirical_risk(model, X, Y):
    model.eval()
    with th.no_grad():
        outputs = model(X)
        criterion = th.nn.BCELoss()
        loss = criterion(outputs, Y)
    return loss.item()
def error_rate(model, X, Y, threshold=0.5):
    model.eval()
    with th.no_grad():
        probs = model(X)
        preds = (probs > threshold).float()
        errors = (preds != Y).float()
    return errors.mean().item()

def cross_validate(model, X, Y, loss_fn, k, seed=0):
    """
    model: an nn.Module instance (will be deep-copied per fold)
    data:  (X, Y) where X is [N, ...], Y is [N, ...] (or [N] for CE)
    k: number of folds (k can be N for LOOCV)
    Returns: (train_losses, val_losses) lists of floats, length k
    """
    th.manual_seed(seed)
    learning_rate = 0.001
    num_epochs = 100
    N = X.size(0)
    if not (1 <= k <= N):
        raise ValueError(f"k must be in [1, N], got k={k}, N={N}")
    # Make folds that cover all samples exactly once (works even if N % k != 0)
    indices = th.randperm(N, device=X.device) 
    folds = th.chunk(indices, k) # Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
    tr_errs, val_errs = [], []
    for i in range(k):
        val_idx = folds[i] 
        train_idx = th.cat([folds[j] for j in range(k) if j != i], dim=0) # Join non validation samples
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_va, Y_va = X[val_idx], Y[val_idx]

        # Fresh model for this fold
        m = copy.deepcopy(model)
        m.train()
        if th.cuda.is_available():
            m = m.cuda()
            X_tr = X_tr.cuda()
            Y_tr = Y_tr.cuda()
            X_va = X_va.cuda()
            Y_va = Y_va.cuda()
        # Minimal default choices
        opt = th.optim.SGD(m.parameters(), lr=1e-2)

        # train exactly like your original loop
        for epoch in range(num_epochs):
            m.train()
            opt.zero_grad()
            outputs = m(X_tr)
            loss = loss_fn(outputs, Y_tr)
            loss.backward()
            opt.step()

        tr_err = error_rate(m,X_tr, Y_tr)
        val_err = error_rate(m,X_va, Y_va)
        # print(f"Fold {i+1}/{k}, val loss = {val_loss:.4f}")
        val_errs.append(val_err)
        tr_errs.append(tr_err)
        if i == k - 1:
            trained_model = m  # keep last fold model
            del X_tr, Y_tr, X_va, Y_va
            th.cuda.empty_cache()
        else:
            del m
            del opt
            th.cuda.empty_cache()

    avgValErrs = sum(val_errs) / k
    avgTrainErrs = sum(tr_errs) / k
    # print(f"seed = {seed}, avg tr error = {avgTrainErrs:.4f}, avg val error = {avgValErrs:.4f}")
    return trained_model, avgValErrs, avgTrainErrs

def create_and_train_model(X, Y, model, seed=42):
    loss_fn = th.nn.BCELoss()
    trained_model, avgValErrs, avgTrainErrs = cross_validate(model, X, Y, loss_fn, k=5, seed=seed)
    return trained_model, avgValErrs, avgTrainErrs

def randomSeedSequence(numSeeds, rngSeed =42):
    random.seed(rngSeed)
    return [random.randint(0, 100000) for _ in range(numSeeds)]



parser = argparse.ArgumentParser()
parser.add_argument("--numSeeds", type=int, default=10)
parser.add_argument("--rngSeed", type=int, default=42)
parser.add_argument("--flushEvery", type=int, default=50)
args = parser.parse_args()

numSeeds = args.numSeeds
rngSeed = args.rngSeed
flushEvery = args.flushEvery

# Data
N_train = 1000
N_test = 1000
d = 10

X_train = th.rand(N_train, d) * 2 - 1
Y_train = th.randint(0, 2, (N_train, 2)).float()

X_test = th.rand(N_test, d) * 2 - 1
Y_test = th.randint(0, 2, (N_test, 2)).float()

seeds = randomSeedSequence(numSeeds, rngSeed=rngSeed)
hidden_size = 5

# Output file (append + periodic flush)
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, f"results_rngSeed_{rngSeed}.csv")

file_exists = os.path.exists(csv_path)
f = open(csv_path, "a", newline="")
writer = csv.writer(f)
if not file_exists:
    writer.writerow(["rngSeed", "seed", "train_error", "val_error", "test_error"])
    f.flush()
    os.fsync(f.fileno())

trainErrs, valErrs, testErrs = [], [], []
buffer = []

print(f"Started program {rngSeed}, numSeeds={numSeeds}...")

for i, seed in enumerate(seeds, start=1):
    model = SimpleFFNN(input_size=d, hidden_size=hidden_size, output_size=2)
    trainedModel, avgCVValErr, avgCVTrainErr = create_and_train_model(
        X_train, Y_train, model, seed=seed
    )

    # test error on same device as model
    device = next(trainedModel.parameters()).device
    testErr = error_rate(trainedModel, X_test.to(device), Y_test.to(device))

    trainErrs.append(avgCVTrainErr)
    valErrs.append(avgCVValErr)
    testErrs.append(testErr)

    buffer.append([rngSeed, seed, avgCVTrainErr, avgCVValErr, testErr])

    # flush every flushEvery or at the end
    if (i % flushEvery == 0) or (i == numSeeds):
        writer.writerows(buffer)
        f.flush()
        os.fsync(f.fileno())
        buffer.clear()
        print(f" Program {rngSeed} Flushed {i}/{numSeeds} rows to {csv_path}")

f.close()
print(f"Program {rngSeed} Completed. Saved results to {csv_path}")