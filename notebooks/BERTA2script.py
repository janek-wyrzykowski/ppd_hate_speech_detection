import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time, datetime, random, gc, traceback
from tqdm import tqdm


def BERTAify(clean_df, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    for comment in clean_df["comment_text"]:
        encoded = tokenizer.encode_plus(
            str(comment),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(clean_df["toxic_level"].values, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():

    # ============================================================
    # CONFIG
    # ============================================================
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    data_train = "../data/df_train.csv"
    data_val = "../data/df_val.csv"
    data_test = "../data/df_test.csv"

    batch_size = 32
    max_length = 64
    epochs = 3
    lr = 2e-5

    # ============================================================
    # LOAD DATA
    # ============================================================
    clean_train = pd.read_csv(data_train)
    clean_val = pd.read_csv(data_val)
    clean_test = pd.read_csv(data_test)

    for df in (clean_train, clean_val, clean_test):
        df["comment_text"] = df["comment_text"].astype(str)
        df["toxic_level"] = df["toxic_level"].astype(int)

    train_dataset = BERTAify(clean_train, tokenizer, max_length=max_length)
    val_dataset = BERTAify(clean_val, tokenizer, max_length=max_length)
    test_dataset = BERTAify(clean_test, tokenizer, max_length=max_length)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ============================================================
    # SETUP MODEL
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)

    if device.type == "cuda":
        print("Using CUDA")

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    best_val_acc = -1
    training_stats = []

    total_t0 = time.time()

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    try:
        for epoch_i in range(epochs):

            print(f"\n======== Epoch {epoch_i+1} / {epochs} ========")

            # -----------------------------
            # TRAINING
            # -----------------------------
            model.train()
            total_train_loss = 0
            t0 = time.time()

            train_bar = tqdm(train_loader, desc=f"Training {epoch_i+1}/{epochs}")

            for batch in train_bar:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                optimizer.zero_grad()

                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)
            training_time = format_time(time.time() - t0)

            print(f"  Average training loss: {avg_train_loss:.4f}")
            print(f"  Training time:         {training_time}")

            # -----------------------------
            # VALIDATION
            # -----------------------------
            model.eval()
            t0 = time.time()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            val_bar = tqdm(val_loader, desc=f"Validation {epoch_i+1}/{epochs}")

            for batch in val_bar:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    outputs = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels
                    )

                loss = outputs.loss
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.cpu().numpy()

                total_eval_loss += loss.item()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                nb_eval_steps += 1

                val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            avg_val_accuracy = total_eval_accuracy / nb_eval_steps
            avg_val_loss = total_eval_loss / nb_eval_steps
            validation_time = format_time(time.time() - t0)

            print(f"  Validation accuracy: {avg_val_accuracy:.4f}")
            print(f"  Validation loss:     {avg_val_loss:.4f}")
            print(f"  Validation time:     {validation_time}")

            # Save best model
            # if avg_val_accuracy > best_val_acc:
            #     best_val_acc = avg_val_accuracy
            #     torch.save(model.state_dict(), "best_bert_state_dict.pt")
            #     print(f"  New best model saved (val_acc={best_val_acc:.4f})")

            training_stats.append({
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Validation Accuracy": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time
            })

            # cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception:
        print("\n❌ Exception occurred during training:\n")
        traceback.print_exc()

    print("\nTraining complete.")
    print(f"Total training time: {format_time(time.time() - total_t0)}")
    print("\nTraining stats:")
    for s in training_stats:
        print(s)


    # model.load_state_dict(torch.load("bert128.pt", map_location='cpu'))
    model.eval()

    predictions = []
    train_bar = tqdm(test_dataloader, desc=f"Testing...")
    for batch in train_bar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():        
                output= model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                
                predictions.extend(list(pred_flat))

    with open(f'BERTpred{max_length}.txt', 'w') as f:
        f.write(f"{predictions}")

if __name__ == "__main__":
    main()
