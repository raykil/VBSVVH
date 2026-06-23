import torch
import torch.nn as nn
import torch.nn.functional as F
def distance_corr(
        var_1:torch.tensor,
        var_2:torch.tensor,
        normedweight:torch.tensor,
        power=1,
        )->torch.tensor:
    
    # Normalize the weights
    normedweight = normedweight/torch.sum(normedweight)*len(var_1)
    
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    return dCorr

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, use_batchnorm=True, dropout=0.0):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")

        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

class ABCDModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        learning_rate=1e-3,
        bce_weight=1.0,
        disco_lambda=0.0,
        flavor="single",
        use_batchnorm=True,
        dropout=0.0,
        weight_decay=1e-2,
        label_smoothing=0.0,
        use_lr_scheduler=True,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
    ):
        super().__init__()

        if flavor not in {"single", "double"}:
            raise ValueError("flavor must be either 'single' or 'double'")

        self.flavor = flavor
        self.bce_weight = bce_weight
        self.disco_lambda = disco_lambda
        self.label_smoothing = label_smoothing

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr

        # ---- model ----
        if flavor == "single":
            self.model = MLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )
        else:
            self.model = nn.ModuleList([
                MLP(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
                MLP(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                ),
            ])

    # ---------------- forward ----------------
    def forward(self, x):
        if self.flavor == "single":
            return self.model(x)

        return torch.stack(
            [self.model[0](x), self.model[1](x)],
            dim=1
        )

    # ---------------- loss ----------------
    def compute_loss(self, batch):
        data, constraint_data, labels, weights = batch

        logits = self(data)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        # label smoothing
        smoothed_labels = labels.float()
        if self.label_smoothing > 0:
            smoothed_labels = (
                smoothed_labels * (1.0 - self.label_smoothing)
                + 0.5 * self.label_smoothing
            )

        scores = torch.sigmoid(logits)

        safe_weights = torch.clamp(weights, min=0.0)
        safe_weights = safe_weights / (torch.mean(safe_weights) + 1e-12)

        # BCE over heads
        bce_components = []
        for h in range(logits.shape[1]):
            bce_components.append(
                F.binary_cross_entropy_with_logits(
                    logits[:, h],
                    smoothed_labels,
                    weight=safe_weights,
                )
            )
        bce = torch.stack(bce_components).sum()

        # ---------------- disco term ----------------
        bkg_mask = labels < 0.5
        disco_term = torch.zeros((), device=logits.device)

        if bkg_mask.sum() > 0:
            bkg_scores = scores[bkg_mask]
            bkg_constraint = constraint_data[bkg_mask, 0]
            bkg_weights = safe_weights[bkg_mask]

            bkg_score_0 = bkg_scores[:, 0]

            if self.flavor == "single":
                if (
                    bkg_score_0.max() - bkg_score_0.min() > 1e-8
                    and bkg_constraint.max() - bkg_constraint.min() > 1e-8
                ):
                    disco_term = distance_corr(
                        bkg_score_0,
                        bkg_constraint,
                        bkg_weights,
                        power=2,
                    )
            else:
                bkg_score_1 = bkg_scores[:, 1]

                if (
                    bkg_score_0.max() - bkg_score_0.min() > 1e-8
                    and bkg_score_1.max() - bkg_score_1.min() > 1e-8
                ):
                    disco_term = distance_corr(
                        bkg_score_0,
                        bkg_score_1,
                        bkg_weights,
                        power=2,
                    )

        total_loss = self.bce_weight * bce + self.disco_lambda * disco_term

        return total_loss, bce, disco_term, bce_components
