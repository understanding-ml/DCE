from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from models.mlp import BlackBoxModel
from models.svm import LinearSVM
from models.lr import LogisticRegression
from models.rbf import RBFNet
from explainers.model import Model

# Variable passed from external: model_name, options: 'mlp', 'svm', 'lr', 'rbf', 'random_forest', 'lightgbm', 'xgboost'

if model_name == "mlp":
    model_raw = BlackBoxModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_raw.parameters(), lr=0.01)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

    for epoch in range(300):
        outputs = model_raw(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    backend = "pytorch"

elif model_name == "svm":
    model_raw = LinearSVM()
    model_raw.fit(X_train, y_train)
    backend = "sklearn"

elif model_name == "lr":
    model_raw = LogisticRegression()
    model_raw.fit(X_train, y_train)
    backend = "sklearn"

elif model_name == "rbf":
    model_raw = RBFNet(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_raw.parameters(), lr=0.01)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

    for epoch in range(300):
        outputs = model_raw(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    backend = "pytorch"

elif model_name == "random_forest":
    model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    model_raw.fit(X_train, y_train)
    backend = "sklearn"

elif model_name == "lightgbm":
    model_raw = LGBMClassifier(n_estimators=100, random_state=42)
    model_raw.fit(X_train, y_train)
    backend = "lightgbm"

elif model_name == "xgboost":
    model_raw = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model_raw.fit(X_train, y_train)
    backend = "xgboost"

else:
    raise ValueError("Unsupported model name: %s" % model_name)

model = Model(model=model_raw, backend=backend)
