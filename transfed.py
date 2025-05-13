import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numbers
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import average_precision_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

def prepare_eicu_data(dataa, datab, test):

    # Define all features in dataset a and b
    shared_feats = [
        'heartrate', 'respiration', 'noninvasivesystolic', 'noninvasivediastolic',
        'noninvasivemean', 'admissionheight', 'dischargeweight','motor',
        'verbal', 'gender_Unknown', 'ethnicity_Asian', 'gender_Other', 
        'ethnicity_Native American', 'ethnicity_African American', 'ethnicity_Hispanic', 
        'gender_Male', 'gender_Female','unitdischargestatus'
    ]

    acols = [
    'Hct', 'chloride', 'Hgb', 'RBC', 'admissionheight', 'dischargeweight',
    'calcium', 'platelets x 1000', 'MCV', 'bicarbonate', 'RDW', 'AST (SGOT)',
    'ALT (SGPT)', 'total protein', 'alkaline phos.', 'magnesium', '-basos',
    'total bilirubin', '-polys', 'respiration', 'noninvasivesystolic',
    'noninvasivediastolic', 'noninvasivemean', 'intubated', 'vent', 'dialysis',
    'verbal', 'meds', 'urine', 'wbc', 'respiratoryrate', 'motor',
    'ph', 'hematocrit', 'bun', 'bilirubin', 'creatinine', 'heartrate',
    'albumin', 'sodium', 'ethnicity_African American',
    'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic',
    'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female',
    'gender_Male', 'gender_Other', 'gender_Unknown', 'unitdischargestatus'
    ]

    bcols = [
    'BUN', 'potassium', 'WBC x 1000', 'heartrate','respiration','noninvasivesystolic',
    'glucose', 'meanbp', 'admissionheight', 'noninvasivediastolic', 'noninvasivemean',
    'dischargeweight', 'anion gap', 'MCH', 'MCHC', '-lymphs', '-monos', '-eos',
    'fio2', 'observationoffset', 'age','sao2', 'eyes', 'motor', 'pao2', 'pco2',
    'ethnicity_African American','verbal',
    'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic',
    'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female',
    'gender_Male', 'gender_Other', 'gender_Unknown', 'unitdischargestatus'
    ]

    test = test[dataa.columns]

    # Experiment to reduce domain A sample size
    # X_dataa, X_atest, y_dataa, y_atest = train_test_split(dataa.iloc[:, :-1], dataa["unitdischargestatus"], test_size=0.9, random_state=88) # using only 1-test_size of dataa
    # dataa = X_dataa
    # dataa['unitdischargestatus'] = y_dataa.astype('int')

    
    atrain = dataa[acols]
    btrain = datab[bcols]
 

    aind = []
    for col in list(acols[:-1]):
        aind.append(list(test).index(col))
    bind = []
    for col in list(bcols[:-1]):
        bind.append(list(test).index(col))


    return atrain, btrain, test, aind, bind


def prepare_dataloader(atrain, btrain, test):
    # Convert target variables to NumPy arrays
    y_atrain = atrain['unitdischargestatus'].astype('int').to_numpy()  # Convert to NumPy array
    X_atrain = atrain.drop(columns=['unitdischargestatus']).to_numpy()  # Convert features to NumPy array
    y_btrain = btrain['unitdischargestatus'].astype('int').to_numpy()
    X_btrain = btrain.drop(columns=['unitdischargestatus']).to_numpy()

    y_test = test['unitdischargestatus'].astype('int').to_numpy()
    X_test = test.drop(columns=['unitdischargestatus']).to_numpy()

    batch_size = 32

    # Create datasets
    atrain_dataset = EicuDataset(X_atrain, y_atrain)
    weights = torch.tensor(unbalanced_dataset_weights(instances=atrain_dataset, num_classes=2), dtype=torch.float64)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    atrain_dataloader = DataLoader(dataset=atrain_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)

    btrain_dataset = EicuDataset(X_btrain, y_btrain)
    btrain_dataloader = DataLoader(dataset=btrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = EicuDataset(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return atrain_dataloader, btrain_dataloader, test_dataloader

def unbalanced_dataset_weights(instances, num_classes):
    count = [0] * num_classes
    for i in range(len(instances)):
        label = int(instances[i][1].item())  # Convert tensor to integer
        count[label] += 1
        
    class_weight = [0.] * num_classes
    total = float(sum(count))
    
    for i in range(num_classes):
        class_weight[i] = total/float(count[i])
    
    weight = [0] * len(instances)
    
    for idx in range(len(instances)):
        label = int(instances[idx][1].item())  # Convert tensor to integer
        weight[idx] = class_weight[label]
        
    return weight

class EicuDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        feats = self.x[idx].to(DEVICE)
        target = self.y[idx].to(DEVICE)
        return feats, target


def prepare_labeled_b_dataloader(anchors, anchors_labels):
    # Convert data arrays to tensors
    anchors_tensor = torch.from_numpy(anchors).float()

    # Convert label arrays to tensors
    anchors_labels_tensor = torch.from_numpy(anchors_labels).int()

    labeled_b_dataset = labeled_b_Dataset(
        anchors_tensor, 
        anchors_labels_tensor
    )

    # Create the DataLoader
    batch_size = 32  # Adjust as needed
    labeled_b_loader = DataLoader(labeled_b_dataset, batch_size=batch_size, shuffle=True)

    return labeled_b_loader

class labeled_b_Dataset(Dataset):
    def __init__(self, anchors, 
                 anchors_labels=None):
        self.anchors = anchors.to(DEVICE)
        self.anchors_labels = anchors_labels.to(DEVICE) if anchors_labels is not None else None

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]

        if self.anchors_labels is not None:
            anchor_label = self.anchors_labels[idx]
            return anchor, anchor_label
        else:
            return anchor


class MLP(nn.Module):
    def __init__(self, l):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_features=l, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=1)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Dropout layers
        self.dropout = nn.Dropout(p=0.4)  # 40% dropout

    def forward(self, x):
        # x = x.to(DEVICE)
        hidden = self.gelu(self.linear1(x))
        hidden = self.dropout(hidden)
        hidden = self.gelu(self.linear2(hidden))
        hidden = self.dropout(hidden)
        hidden = self.gelu(self.linear3(hidden))
        hidden = self.dropout(hidden)
        out = self.sigmoid(self.linear4(hidden))
        return hidden, out
    


def train_orange(model, optimizer, atrain_dataloader, epoch):
    # train domain A: supervised BCE loss on domain A
    labels = []
    preds  = []
    probs = []
    train_running_loss = 0.0
    
    model.train()
    for index, data in enumerate(atrain_dataloader):
        batch_inputs, batch_labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        _, outputs = model(batch_inputs)
        outputs = outputs.squeeze()
        loss = F.binary_cross_entropy(outputs, batch_labels.float()).mean()
        
        loss.backward()
        optimizer.step()

        # evaluation
        
        model.eval()
        with torch.no_grad():
            _, outputs = model(batch_inputs)
            outputs = outputs.squeeze()
            probs.extend(outputs.cpu().numpy())
            binary_outputs = (outputs > 0.5).float()
            preds.extend(binary_outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
        model.train()
        
        train_running_loss += loss.item()
    
    # Compute metrics
    if len(labels) > 0 and len(preds) > 0:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        correct_preds_count = tn + tp
        total_count = tn + fp + fn + tp
        prauc = average_precision_score(labels, probs)
        
        accuracy = (correct_preds_count / total_count) * 100
        
        print(f'Train Epoch {epoch+1}:\n    Accuracy: {accuracy:.2f}%')
        print("      orange train prauc: ", prauc)
        print()
        
    else:
        print("No data to compute metrics.")
    
    return train_running_loss


def noise(X, noise_level):
    if isinstance(X,torch.Tensor):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=X.size())
        noise = torch.autograd.Variable(torch.FloatTensor(noise).to(X.device))
        return X + noise
    elif isinstance(X,np.ndarray):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
        return X + noise
    elif isinstance(X,numbers.Number):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=1)
        return X + noise
    else:
        raise ValueError('No data to augment')

class FixMatch(nn.Module):
    # FixMatch: Semi-supervised learning
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs_uw, inputs_us, model, ssl_preds, ssl_labels, ssl_mask, y_unlabeled):

        with torch.no_grad():
            hidden_uw, outputs_uw = model(inputs_uw)
            probs_unlabeled = outputs_uw.squeeze()
            neg_outputs = 1 - outputs_uw
            probs = torch.stack((neg_outputs, outputs_uw), dim=1)
            max_p, p_hat = torch.max(probs, dim=1)


        mask = max_p.ge(self.threshold).float()
        if mask.sum() > 0:
            hidden_us, outputs_us = model(inputs_us)
            # pseudo-labeling loss
            ssl_loss = (F.binary_cross_entropy(outputs_us.squeeze(), p_hat.squeeze().float()) * mask).mean()
            dif = hidden_uw - hidden_us
            # hidden representation consistency loss
            consistency_regularization_loss = torch.norm(dif, p='fro') / len(inputs_uw)
            ssl_preds.append(outputs_us.squeeze())
            ssl_labels.append(y_unlabeled.squeeze())
            ssl_mask.append(mask.detach())
        else:
            ssl_loss = torch.tensor(0.0, device=DEVICE)
            consistency_regularization_loss = torch.tensor(0.0, device=DEVICE)

        return ssl_loss, consistency_regularization_loss


def compute_ssl(model, optimizer, epoch, x_labeled, y_labeled, x_unlabeled, lambda_u, labeldp, score, ssl_obj, ssl_preds, ssl_labels, ssl_mask, y_unlabeled, start, threshold=0.95):
    # compute all ssl losses
    
    _, logits_labeled = model(x_labeled)
    cls_loss = F.binary_cross_entropy(logits_labeled.squeeze(), y_labeled.float()).mean()
    
    optimizer.zero_grad()
    cls_loss.backward()
    optimizer.step()
    
    ssl_loss = torch.tensor(0.0, device=DEVICE)
    consistency_regularization_loss = torch.tensor(0.0, device=DEVICE)
    
    if lambda_u != 0:
        # data augmentation
        inputs_uw = noise(X=x_unlabeled, noise_level=0.05).to(DEVICE)
        inputs_us = noise(X=x_unlabeled, noise_level=0.2).to(DEVICE)
        optimizer.zero_grad()
        
        # then compute pseudo-labeling loss and hidden representation consistency loss
        ssl_loss, consistency_regularization_loss = ssl_obj(inputs_uw=inputs_uw, inputs_us=inputs_us, model=model, ssl_preds=ssl_preds, ssl_labels=ssl_labels, ssl_mask=ssl_mask, y_unlabeled=y_unlabeled)
        # ssl_labels and y_unlabeled used for evaluation

        if ssl_loss != 0:
            lambda_cons = 0.0001
            loss = lambda_u * ssl_loss + lambda_cons * consistency_regularization_loss
            loss.backward()
            optimizer.step()
    
    return cls_loss.item() + lambda_u * ssl_loss.item(), consistency_regularization_loss.item(), lambda_u * ssl_loss.item()

def train_purple(model, optimizer, shared_loader, unlabeled_loader, epoch, lambda_u, labeldp, score, ssl_obj, start):
    # train domain B

    labels = []
    preds  = []
    probs = []
    total_loss = 0.0
    label_iter = iter(shared_loader)
    unlabel_iter = iter(unlabeled_loader)
    len_iter = max(len(unlabel_iter), len(label_iter))

    ssl_preds = []
    ssl_labels = []
    ssl_mask = []
    cons_loss = 0.0
    ppssl_loss = 0.0
    for i in range(len_iter):
        try:
            x_unlabeled, y_unlabeled = next(unlabel_iter)
        except StopIteration:
            unlabel_iter = iter(unlabeled_loader)
            x_unlabeled, y_unlabeled = next(unlabel_iter)

        try:
            anchor_batch, anchor_labels_batch = next(label_iter)
        except StopIteration:
            label_iter = iter(shared_loader)
            anchor_batch, anchor_labels_batch = next(label_iter)

        loss, consistency_regularization_loss, ssl_loss = compute_ssl(
            model=model, 
            optimizer=optimizer, 
            epoch=epoch, 
            x_labeled=anchor_batch, 
            y_labeled=anchor_labels_batch, 
            x_unlabeled=x_unlabeled, 
            lambda_u=lambda_u, 
            labeldp=labeldp, 
            score=score, 
            ssl_obj=ssl_obj, 
            ssl_preds=ssl_preds, 
            ssl_labels=ssl_labels, 
            ssl_mask=ssl_mask, 
            y_unlabeled=y_unlabeled, 
            start=start
        )
        cons_loss += consistency_regularization_loss
        ppssl_loss += ssl_loss

        # evaluation
        model.eval()
        with torch.no_grad():
            _, outputs = model(anchor_batch)
            outputs = outputs.squeeze()
            probs.extend(outputs.cpu().numpy())
            binary_outputs = (outputs >= 0.5).float()
            preds.extend(binary_outputs.cpu().numpy())
            labels.extend(anchor_labels_batch.cpu().numpy())
        model.train()

        total_loss += loss

    # Epoch SSL stats
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    nsamples = 0
    if len(ssl_preds) > 0:
        epoch_predictions = torch.cat(ssl_preds).squeeze()
        epoch_labels = torch.cat(ssl_labels).squeeze()
        epoch_mask = torch.cat(ssl_mask).squeeze()
        masked_predictions = epoch_predictions[epoch_mask.bool()]
        masked_labels = epoch_labels[epoch_mask.bool()]
        masked_binary_predictions = (masked_predictions >= 0.5).float()
        correct_preds = (masked_binary_predictions == masked_labels).float()
        accuracy = correct_preds.mean().item()

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        true_positives = ((masked_binary_predictions == 1) & (masked_labels == 1)).sum().float()
        false_positives = ((masked_binary_predictions == 1) & (masked_labels == 0)).sum().float()
        false_negatives = ((masked_binary_predictions == 0) & (masked_labels == 1)).sum().float()

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-8)  # Avoid division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        nsamples = len(masked_predictions)

    if len(labels) > 0 and len(preds) > 0:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        prauc = average_precision_score(labels, probs)

        correct_preds_count = tn + tp
        total_count = tn + fp + fn + tp
        avg_loss = total_loss / len_iter
        if nsamples != 0:
            ppssl_loss = ppssl_loss / nsamples

        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        print("      train acc: ",  (correct_preds_count / total_count) * 100)
        print("      purple train prauc: ", prauc)

    else:
        print("No data to compute metrics.")

    return total_loss



class SupConLoss(nn.Module):
    # Supervised contrastive loss
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device  # Automatically get device from input features
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [batch_size, feature_dim]')
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).to(device)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1  # Single view
        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positives
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss



def train_supcons(model_purple, model_orange, optimizer_purple, optimizer_orange, criterion_contrastive, pairs_labeled_dataloader, btrain_dataloader, atrain_dataloader, lambda_sc, start, epoch):
    # train supervised contrastive loss
    purple_unlabeled_iter = iter(btrain_dataloader)
    purple_iter = iter(pairs_labeled_dataloader)
    orange_iter = iter(atrain_dataloader)
    len_iter = max(len(purple_unlabeled_iter), len(orange_iter))

    total_loss = 0
    ssl_samples_count = 0  # Add counter for SSL samples
    ssl_loss_sum = 0.0    # Add accumulator for SSL losses

    for i in range(len_iter):
        try:
            x_purple, y_purple = next(purple_iter)
        except StopIteration:
            purple_iter = iter(pairs_labeled_dataloader)
            x_purple, y_purple= next(purple_iter)
        try:
            x_orange, y_orange = next(orange_iter)
        except StopIteration:
            orange_iter = iter(atrain_dataloader)
            x_orange, y_orange = next(orange_iter)
        

        x_purple = x_purple.to(DEVICE)
        y_purple = y_purple.to(DEVICE)
        x_orange = x_orange.to(DEVICE)
        y_orange = y_orange.to(DEVICE)

        # Zero gradients
        optimizer_purple.zero_grad()
        optimizer_orange.zero_grad()

        hidden_purple, logits_purple = model_purple(x_purple)
        hidden_orange, logits_orange = model_orange(x_orange)

        # Normalize embeddings
        embeddings_purple_norm = F.normalize(hidden_purple, p=2, dim=1)
        embeddings_orange_norm = F.normalize(hidden_orange, p=2, dim=1)

        embeddings = torch.cat([embeddings_purple_norm, embeddings_orange_norm], dim=0)  # [2*batch_size, embedding_dim]
        labels = torch.cat([y_purple, y_orange], dim=0)  # [2*batch_size]

        loss_contrastive = criterion_contrastive(embeddings, labels)

        supcons_loss = lambda_sc * loss_contrastive

        supcons_loss.backward()
        optimizer_purple.step()
        optimizer_orange.step()

        total_loss += supcons_loss.item()

    if epoch > start:
        len_iter = max(len(purple_unlabeled_iter), len(orange_iter))
        # Add adaptive threshold that starts lower and increases over time
        # confidence_threshold = min(0.7 + epoch * 0.01, 0.9)  # Gradually increase from 0.7 to 0.9
        confidence_threshold = 0.9
        for i in range(len_iter):
            try:
                x_purple, _ = next(purple_unlabeled_iter)
            except StopIteration:
                purple_unlabeled_iter = iter(btrain_dataloader)
                x_purple, y_purple = next(purple_unlabeled_iter)
            try:
                x_orange, y_orange = next(orange_iter)
            except StopIteration:
                orange_iter = iter(atrain_dataloader)
                x_orange, y_orange = next(orange_iter)

            # Zero gradients
            optimizer_purple.zero_grad()
            optimizer_orange.zero_grad()

            hidden_purple, logits_purple = model_purple(x_purple)
            hidden_orange, logits_orange = model_orange(x_orange)

            probs_unlabeled = logits_purple.squeeze()
            neg_outputs = 1 - probs_unlabeled
            probs = torch.stack((neg_outputs, probs_unlabeled), dim=1)
            max_p, p_hat = torch.max(probs, dim=1)
            mask = max_p.ge(confidence_threshold)  # Use adaptive threshold

            if mask.float().sum() > 0:
                # Track SSL metrics
                num_selected = mask.float().sum().item()
                ssl_samples_count += num_selected

                # Normalize embeddings
                embeddings_purple_norm = F.normalize(hidden_purple, p=2, dim=1)
                embeddings_orange_norm = F.normalize(hidden_orange, p=2, dim=1)

                # Apply the mask to select high-confidence embeddings from Purple
                mask = mask.squeeze() 
                selected_embeddings_purple = embeddings_purple_norm[mask]  # Shape: [num_selected_purple, embedding_dim]


                embeddings = torch.cat([selected_embeddings_purple, embeddings_orange_norm], dim=0)  # [2*batch_size, embedding_dim]
                labels = torch.cat([p_hat[mask].view(-1), y_orange.view(-1)], dim=0)  # Shape: [num_selected_purple + 32]

                loss_contrastive = criterion_contrastive(embeddings, labels)

                supcons_loss = lambda_sc * loss_contrastive
                
                # Track SSL loss
                ssl_loss_sum += supcons_loss.item() * num_selected

                supcons_loss.backward()
                optimizer_purple.step()
                optimizer_orange.step()

                total_loss += supcons_loss.item()

    # Calculate average SSL loss per sample
    avg_ssl_loss = ssl_loss_sum / ssl_samples_count if ssl_samples_count > 0 else 0.0


    return total_loss




        






def train(model_purple, model_orange, optimizer_purple, optimizer_orange, purple_scheduler, orange_scheduler, atrain_dataloader, btrain_dataloader, test_dataloader, aind, bind, pairs_labeled_dataloader, epochs, lambda_sc, lambda_u, labeldp, start):
    # overall training procedure

    ssl_obj = FixMatch(threshold=0.95).to(DEVICE)

    score = np.zeros(2) + 0.98

    criterion_contrastive = SupConLoss(temperature=0.07).to(DEVICE)
    
    for epoch in range(epochs):
        model_purple.train()
        model_orange.train()

        # Track average losses per epoch
        # train domain A
        avg_orange_loss = train_orange(model=model_orange, optimizer=optimizer_orange, atrain_dataloader=atrain_dataloader, epoch=epoch)
        avg_orange_loss = avg_orange_loss / len(atrain_dataloader)  # Average over batches
        
        # train domain B
        avg_purple_loss = train_purple(model=model_purple, optimizer=optimizer_purple, shared_loader=pairs_labeled_dataloader, unlabeled_loader=btrain_dataloader, epoch=epoch, lambda_u=lambda_u, labeldp=labeldp, score=score, ssl_obj=ssl_obj, start=start)
        avg_purple_loss = avg_purple_loss / (len(pairs_labeled_dataloader) + len(btrain_dataloader))  # Average over batches
        
        # train supervised contrastive loss
        total_loss = avg_orange_loss + avg_purple_loss
        avg_supcons_loss = 0
        if lambda_sc > 0:
            current_lambda_sc = lambda_sc * min(1.0, epoch / 15.0)  # Gradual warmup over 10 epochs
            avg_supcons_loss = train_supcons(model_purple, model_orange, optimizer_purple, optimizer_orange, criterion_contrastive, pairs_labeled_dataloader, btrain_dataloader, atrain_dataloader, current_lambda_sc, start, epoch)
            avg_supcons_loss = avg_supcons_loss / (len(pairs_labeled_dataloader) + len(btrain_dataloader))  # Average over batches
            total_loss += avg_supcons_loss
        
            
        print(f"Epoch {epoch} losses:")
        print(f"Orange loss: {avg_orange_loss:.4f}")
        print(f"Purple loss: {avg_purple_loss:.4f}")
        print(f"SupCon loss: {avg_supcons_loss:.4f}")
        print(f"Total loss: {total_loss:.4f}")

        
        # Evaluation on Purple Model
        test_running_loss = 0.0
        labels = []
        preds  = []
        probs = []
        model_purple.eval()
        for index, data in enumerate(test_dataloader):
            
            batch_inputs, batch_labels = data[0][:,bind].to(DEVICE).type(torch.float), data[1].to(DEVICE).type(torch.float)

            _, outputs = model_purple(batch_inputs)
            outputs = outputs.squeeze()

            loss    = F.binary_cross_entropy(outputs, batch_labels).mean()
            probs.extend(outputs.detach().cpu().numpy())
            binary_outputs = (outputs > 0.5).float()
            preds.extend(binary_outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

            test_running_loss += loss.item()


        purple_scheduler.step(test_running_loss / len(test_dataloader))
        print("purple lr: ", optimizer_purple.param_groups[0]['lr'])
        if len(labels) >= 2 and len(set(labels)) > 1:  # Ensure confusion_matrix can ravel
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

            correct_preds_count = tn + tp
            total_count         = tn + fp + fn + tp
            se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp + fp + fn) > 0 else 0.0
            prauc = average_precision_score(labels, probs) if len(set(labels)) > 1 else 0.0
            print("               purple test loss: ", test_running_loss / len(test_dataloader))
            print("               purple test acc: ",  (correct_preds_count / total_count) * 100)
            print("               purple se = ", se)
            print("               purple pp = ", pp)
            print("                                  purple test F-score: ", f1)
            print("                                  purple test prauc: ", prauc)

        else:
            print("Insufficient data to compute confusion matrix for Purple Model.")

        # Evaluation on Orange Model
        test_running_loss = 0.0
        labels = []
        preds  = []
        probs = []
        model_orange.eval()
        for index, data in enumerate(test_dataloader):
            
            batch_inputs, batch_labels = data[0][:,aind].to(DEVICE).type(torch.float), data[1].to(DEVICE).type(torch.float)

            _, outputs = model_orange(batch_inputs)
            outputs = outputs.squeeze()

            loss    = F.binary_cross_entropy(outputs, batch_labels).mean()
            
            probs.extend(outputs.detach().cpu().numpy())
            binary_outputs = (outputs > 0.5).float()
            preds.extend(binary_outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

            test_running_loss += loss.item()
    

        orange_scheduler.step(test_running_loss / len(test_dataloader))
        print("orange lr: ", optimizer_orange.param_groups[0]['lr'])
        if len(labels) >= 2 and len(set(labels)) > 1:  # Ensure confusion_matrix can ravel
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()


            correct_preds_count = tn + tp
            total_count         = tn + fp + fn + tp
            se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp + fp + fn) > 0 else 0.0
            prauc = average_precision_score(labels, probs) if len(set(labels)) > 1 else 0.0
            print("              orange test loss: ", test_running_loss / len(test_dataloader))
            print("              orange test acc: ",  (correct_preds_count / total_count) * 100)
            print("              orange se = ", se)
            print("              orange pp = ", pp)
            print("                                   orange test F-score: ", f1)
            print("                                   orange test prauc: ", prauc)

        else:
            print("Insufficient data to compute confusion matrix for Orange Model.")

def main():

    set_seed(88)

    parser = argparse.ArgumentParser(description='Training script with parameter tuning.')
    parser.add_argument('--lambda_sc', type=float, required=True, help='Value for lambda_sc, weight for SupCon loss')
    parser.add_argument('--lambda_u', type=float, required=True, help='Value for lambda_u, weight for ssl loss')
    args = parser.parse_args()

    lambda_sc = args.lambda_sc
    lambda_u = args.lambda_u
    start = 0

    

    purple_lr = 3e-5
    orange_lr = 5e-5
    patience = 5
    epochs = 40
    labeldp = False


    config = {
    "orange_lr": orange_lr,
    "purple_lr": purple_lr,
    "patience": patience,
    "batch_size": 32,
    "epochs": epochs,
    "lambda_sc": lambda_sc,
    "lambda_u": lambda_u,
    "labeldp": labeldp,
    "start":start,
    "notes": ""
    }


    print("preparing data")
    dataa = pd.read_csv(filepath_or_buffer="dataa.csv")
    datab = pd.read_csv(filepath_or_buffer="datab_unlabeled.csv")
    test = pd.read_csv(filepath_or_buffer="midwest_test.csv")

    # Load data and labels from the .npz file
    pairs_labeled = np.load('anchors_labeled.npz')

    anchors_labeled = pairs_labeled['anchors']
    anchors_labels_labeled = pairs_labeled['anchors_labels']


    print("labeled samples: ", len(anchors_labeled))



    atrain, btrain, test, aind, bind = prepare_eicu_data(dataa, datab, test)
    pairs_labeled_dataloader = prepare_labeled_b_dataloader(anchors_labeled, anchors_labels_labeled)



    print(len(bind))
    print("preparing dataloaders")
    atrain_dataloader, btrain_dataloader, test_dataloader = prepare_dataloader(atrain, btrain, test)
    print("run ftl")

    
    model_purple = MLP(l=len(bind)).to(DEVICE)
    print(btrain.shape[1]-1)
    optimizer_purple = torch.optim.Adam(params=model_purple.parameters(), lr=purple_lr)
    purple_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_purple, 
                                                       mode='min', 
                                                       factor=0.5, 
                                                       patience=patience, 
                                                       verbose=True)
    model_orange = MLP(l=len(aind)).to(DEVICE)
    print(atrain.shape[1]-1)
    optimizer_orange = torch.optim.Adam(params=model_orange.parameters(), lr=orange_lr)
    orange_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_orange, 
                                                       mode='min', 
                                                       factor=0.5, 
                                                       patience=patience, 
                                                       verbose=True)
    
    
    train(model_purple, model_orange, optimizer_purple, optimizer_orange, purple_scheduler, orange_scheduler, atrain_dataloader, btrain_dataloader, test_dataloader, aind, bind, pairs_labeled_dataloader, epochs, lambda_sc, lambda_u, labeldp, start)
    



if __name__ == '__main__':
    main()
