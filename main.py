import os
# 🔹 Python standard library for working with the filesystem and paths.
# 🔸 Why is it needed? To check for the existence of the saved model file (best_combined.pt), and other file/path operations.

import sys
# 🔹 Access to system parameters (platform, argv, exit).
# 🔸 Why is it needed? To detect the operating system (Windows/macOS/Linux) and exit the program safely (sys.exit).

import random
# 🔹 Python-level random number generator.
# 🔸 Why is it needed? To set the seed and ensure reproducibility (get the same result everywhere).

import subprocess
# 🔹 Run OS processes (external programs like notepad.exe and calc.exe).
# 🔸 Why is it needed? For actions like opening Notepad/Calculator natively.

import webbrowser
# 🔹 Open a URL in the system’s default browser.
# 🔸 Why is it needed? For actions like opening Google/YouTube or web fallbacks.

import time
# 🔹 Time functions (sleep, etc.).
# 🔸 Why is it needed? To pause a bit before exiting (let TTS finish speaking).

import numpy as np
# 🔹 Efficient arrays and numerical computations.
# 🔸 Why is it needed? To collect predictions/labels and simple vector operations.

import torch
# 🔹 PyTorch core (tensors, GPU/CPU ops, autograd).
# 🔸 Why is it needed? To build/train/evaluate the deep learning model.

from torch.utils.data import Dataset, DataLoader
# 🔹 Dataset: base class for defining a custom dataset.
# 🔹 DataLoader: mini-batching, shuffling, and expanding across multiple workers (if needed).
# 🔸 Why is it needed? We create a custom Dataset (IntentDataset) and use DataLoader for batching.

from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
# 🔹 BertTokenizer: WordPiece tokenizer aligned with BERT-base-uncased.
# 🔹 BertForSequenceClassification: BERT with a ready classification head (on the [CLS] token).
# 🔹 get_linear_schedule_with_warmup: LR scheduling with initial warmup and linear decay.
# 🔸 Why is it needed? Standard implementation of fine-tuning BERT on sentence classification.

from torch.optim import AdamW
# 🔹 Adam optimizer with corrected Weight Decay (the version suitable for transformers).
# 🔸 Why is it needed? For better stability and performance of BERT compared to vanilla Adam.

from sklearn.metrics import accuracy_score, f1_score
# 🔹 Evaluation metrics: Accuracy (simple) and weighted F1 (better for imbalanced classes).
# 🔸 Why is it needed? To report performance on Val/Test and compare models.

from sklearn.model_selection import train_test_split
# 🔹 Random split into Train/Validation/Test with optional stratify.
# 🔸 Why is it needed? Fair evaluation and prevention of leakage.

from datasets import load_dataset
# 🔹 Load Hugging Face datasets in one line.
# 🔸 Why is it needed? Access to SNIPS (built_in_intents version in this code).

import pyttsx3
# 🔹 Offline TTS engine (no internet).
# 🔸 Why is it needed? Voice responses to the user (speak).

import speech_recognition as sr
# 🔹 Speech recognition library with various backends; here we use Google.
# 🔸 Why is it needed? Get voice input from the microphone and convert to text.

from datetime import datetime
# 🔹 System date/time.
# 🔸 Why is it needed? Tell the current time (tell_time).

# -----------------------------
# 0. Configuration
# -----------------------------
SEED = 42
# 🔹 Fixed seed for reproducibility. 42 is a common choice (arbitrary but fixed).
# 🔸 If changed, shuffle order and initialization change and results may shift slightly.

BATCH_SIZE = 16
# 🔹 Mini-batch size. Trade-off of speed/memory: larger → faster processing but higher RAM/VRAM use.
# 🔸 On CPU or a weaker GPU, 16 is a safe choice. On stronger GPUs you can try 32/64.

EPOCHS = 3
# 🔹 Number of full passes over the training data.
# 🔸 A small value (3) is good for a quick demo; if validation is steady/high and no overfitting is observed, you can increase it.

LR = 2e-5
# 🔹 Learning rate. For fine-tuning BERT, 2e-5 or 3e-5 are standard.
# 🔸 Higher (e.g., 1e-4) risks oscillation/instability; lower (1e-5) is slower but more stable.

MAX_LEN = 64
# 🔹 Max token length for padding/truncation.
# 🔸 SNIPS has short sentences; 64 is enough. Increasing it → more computation/memory.

CONFIDENCE_THRESHOLD = 0.40  # ML fallback confidence threshold
# 🔹 Confidence threshold for executing the model’s output. If lower than this → ask the user to repeat.
# 🔸 Why 0.4? A compromise between “executing commands with certainty” and “avoiding wrong execution in ASR noise.”

# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed=SEED):
    # 🔹 Goal: reproducibility. Set all sources of randomness to a single seed.
    random.seed(seed)              # Python-level randomness
    np.random.seed(seed)           # NumPy randomness
    torch.manual_seed(seed)        # PyTorch core on CPU
    torch.cuda.manual_seed_all(seed)  # On GPU (if multiple cards exist)

set_seed()
# 🔹 This call ensures random operations (shuffle, init, …) are as consistent as possible.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 🔹 Use GPU if CUDA is available; otherwise CPU.
# 🔸 GPU usually makes training much faster. This line only performs the selection.

print("Using device:", device)
# 🔹 Log for transparency: know which hardware the program is running on.

# -----------------------------
# 2. Text-to-Speech
# -----------------------------
def speak(text):
    # 🔹 Convert text to speech with pyttsx3 (offline and simple).
    try:
        engine = pyttsx3.init()  # Initialize engine; on some systems it may require driver/voice.
        engine.say(text)         # Queue text for speech.
        engine.runAndWait()      # Execute speaking and wait to finish so it’s not cut off.
    except Exception as e:
        # 🔹 If TTS fails (e.g., voice not installed), at least print the text so output isn’t lost.
        print("[TTS error]", e)
        print("Assistant:", text)

# -----------------------------
# 3. OS Helpers & Actions
# -----------------------------
def _is_windows(): return sys.platform.startswith("win")
# 🔹 Helper: Is the OS Windows? Some actions are platform-dependent (notepad.exe, calc.exe).

def _is_mac(): return sys.platform == "darwin"
# 🔹 Helper: Is it macOS? For opening TextEdit/Calculator on Mac.

def open_browser():
    # 🔹 Sample action: open the browser to Google.
    webbrowser.open("https://www.google.com")
    speak("I opened the browser for you")
    # 🔸 webbrowser saves us from needing program paths (cross-platform)

def tell_time():
    # 🔹 Sample action: announce the current system time.
    current_time = datetime.now().strftime("%H:%M")
    speak(f"The time is {current_time}")
    # 🔸 strftime with HH:MM formats the time display.

def open_notepad():
    # 🔹 Sample action: open a native text editor (with web fallback).
    try:
        if _is_windows():
            subprocess.Popen(["notepad.exe"])
        elif _is_mac():
            subprocess.Popen(["open", "-a", "TextEdit"])
        else:
            # 🔸 On Linux, depending on the desktop, different editors may be installed:
            for cmd in (["gedit"], ["xed"], ["kate"], ["leafpad"], ["mousepad"]):
                try:
                    subprocess.Popen(cmd)  # Try launching each
                    break
                except Exception:
                    continue
            else:
                # 🔸 If none found, open a web editor so the action isn’t empty.
                webbrowser.open("https://docs.new")
        speak("Opened a text editor.")
    except Exception:
        # 🔹 Any other error → web fallback.
        webbrowser.open("https://docs.new")
        speak("I opened a web editor instead.")

def open_calculator():
    # 🔹 Sample action: open a native calculator (with web fallback).
    try:
        if _is_windows():
            subprocess.Popen(["calc.exe"])
        elif _is_mac():
            subprocess.Popen(["open", "-a", "Calculator"])
        else:
            for cmd in (["gnome-calculator"], ["kcalc"], ["qalculate-gtk"], ["xcalc"]):
                try:
                    subprocess.Popen(cmd)
                    break
                except Exception:
                    continue
            else:
                webbrowser.open("https://www.google.com/search?q=calculator")
        speak("Calculator opened.")
    except Exception:
        webbrowser.open("https://www.google.com/search?q=calculator")
        speak("I opened a web calculator.")

def play_youtube():
    # 🔹 Sample action: open YouTube.
    webbrowser.open("https://www.youtube.com")
    speak("I opened YouTube for you")

def say_hello():
    # 🔹 Sample action: greet.
    speak("Hello! How can I help you?")

def say_goodbye():
    # 🔹 Sample action: say goodbye and exit the program safely.
    speak("Goodbye! See you later.")
    time.sleep(0.5)  # 🔸 Short pause so the audio fully plays.
    sys.exit(0)      # 🔸 Exit.

# -----------------------------
# 4. Load SNIPS + add custom intents
# -----------------------------
print("Loading SNIPS dataset...")
# 🔹 Inform the user/logs that the dataset is being loaded.

snips = load_dataset("snips_built_in_intents")
# 🔹 Load the SNIPS dataset (built_in_intents version) from Hugging Face Datasets.
# 🔸 This set has columns 'text' (sentence) and 'label' (class ID).

snips_texts = list(snips["train"]["text"])
# 🔹 Extract the list of training sentences from the train split.
snips_labels = list(snips["train"]["label"])
# 🔹 Extract the list of numeric labels aligned with the sentences.

snips_label_names = snips["train"].features["label"].names
# 🔹 Readable names for labels (ClassLabel.names) — useful for reporting/mappings.
print("SNIPS label names:", snips_label_names)
# 🔹 Print class names for clarity.

custom_intent_names = [
    "open_browser_custom",
    "open_notepad_custom",
    "open_calculator_custom",
    "play_youtube_custom",
    "greet_custom",
    "goodbye_custom"
]
# 🔹 Define 6 custom classes for our system actions (beyond classic SNIPS).
# 🔸 Reason: We want the model to be sensitive to phrases related to these actions.

custom_phrases = {
    "open_browser_custom": ["open the browser","please open browser","launch browser","open google","start a web browser"],
    "open_notepad_custom": ["open notepad","open text editor","start notepad","please open notepad","open a text editor"],
    "open_calculator_custom": ["open calculator","start calculator","please open calculator","launch calculator"],
    "play_youtube_custom": ["open youtube","play youtube","play a youtube video","open youtube please","launch youtube"],
    "greet_custom": ["hello","hi","hey","good morning","good evening"],
    "goodbye_custom": ["goodbye","bye","see you","exit","stop assistant"]
}
# 🔹 Sample phrases for each custom intent.
# 🔸 The greater the linguistic variety, the better the model’s generalization. (We put a few base phrases here.)

all_label_names = snips_label_names + custom_intent_names
# 🔹 Final list of class names = SNIPS base classes + custom classes.
# 🔸 Used to set num_labels and for mappings.

label2id = {name: idx for idx, name in enumerate(all_label_names)}
# 🔹 Map class name → numeric ID (0..N-1) for training/prediction.

id2label = {v: k for k, v in label2id.items()}
# 🔹 Reverse mapping: numeric ID → class name (for reporting/action).

texts, labels = [], []
# 🔹 Final arrays from which train/validation/test sets are built.

# SNIPS examples
for t, l in zip(snips_texts, snips_labels):
    texts.append(t)
    labels.append(l)
# 🔹 Add all train examples from SNIPS to the final arrays.

# Custom examples: duplicate each 5x for stratify
for cname in custom_intent_names:
    cid = label2id[cname]
    examples = custom_phrases.get(cname, [])
    for ex in examples:
        for _ in range(5):  # duplicate
            texts.append(ex)
            labels.append(cid)
# 🔹 The custom classes have few samples;
# 🔸 train_test_split with stratify requires each class to have ≥ 2 samples.
# 🔸 Here we repeat each phrase 5 times to satisfy the condition and keep the split stable.
# ⚠️ Analytical note for presentation: This makes the class distribution somewhat artificial (bias). It’s better to report this.

print("Total combined examples:", len(texts), "Total labels:", len(set(labels)))
# 🔹 Log: how many samples we have and how many classes (SNIPS + custom).

# -----------------------------
# 5. Train/Val/Test Split
# -----------------------------
print("Splitting into train/val/test...")
# 🔹 Inform about the split step.

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=SEED, stratify=labels
)
# 🔹 First split: 75% → Train and 25% → Temp, with stratify across all labels.
# 🔸 Why 0.25? So we can later split Temp in half (Val/Test), i.e., 12.5% + 12.5%.
# 🔸 random_state=SEED for reproducibility.

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
)
# 🔹 Second split: split Temp into Val and Test halves (each 12.5% of total).
# 🔸 Again stratify to keep class distributions similar.

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
# 🔹 Print the size of each partition to give an accurate picture during presentation.

# -----------------------------
# 6. Dataset class + Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 🔹 Tokenizer aligned with BERT-base-uncased (lowercasing/WordPiece).
# 🔸 Why this version? Because our BERT model expects this vocabulary/preprocessing.

class IntentDataset(Dataset):
    # 🔹 Custom PyTorch dataset: converts raw text to model inputs using the tokenizer.
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        # 🔸 List of strings (sentences)
        self.labels = labels
        # 🔸 List of integers (class IDs)
        self.tokenizer = tokenizer
        # 🔸 Tokenizer object from Transformers
        self.max_len = max_len
        # 🔸 Token truncation/padding length; should match MAX_LEN.

    def __len__(self): return len(self.texts)
    # 🔹 Number of samples in this dataset (needed by DataLoader).

    def __getitem__(self, idx):
        # 🔹 Convert the i-th sample to the model’s input format and return it.
        text = str(self.texts[idx])     # Ensure it’s a string
        label = int(self.labels[idx])   # Ensure the label is an int
        encoding = self.tokenizer(
            text, add_special_tokens=True, padding="max_length",
            truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        # 🔸 add_special_tokens=True: add [CLS] and [SEP] per BERT.
        # 🔸 padding="max_length": pad all sentences to MAX_LEN (equal-length inputs → simpler).
        # 🔸 truncation=True: cut sentences longer than MAX_LEN (accuracy/perf trade-off).
        # 🔸 return_tensors="pt": convert directly to PyTorch tensors.

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            # 🔹 Token IDs, shape [seq_len]. squeeze(0) removes batch=1 dimension
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # 🔹 Attention mask: 1 for real tokens, 0 for padding (BERT uses this to ignore padding).
            "labels": torch.tensor(label, dtype=torch.long)
            # 🔹 Class label as a LongTensor (required for the model’s internal CrossEntropyLoss).
        }

# 🔹 Build dataset objects for each split:
train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
test_dataset = IntentDataset(test_texts, test_labels, tokenizer)

# 🔹 DataLoaders: mini-batching + shuffle for train (to reduce dependence on data order).
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
# 🔸 Since we use fixed padding (max_length), we don’t need a custom collate_fn.
# 🔸 If we wanted dynamic padding, we’d need a DataCollator or custom collate.

# -----------------------------
# 7. Model setup
# -----------------------------
num_labels = len(all_label_names)
# 🔹 Number of classes (labels) = SNIPS classes + custom action classes.
# 🔸 Why is it needed? To build the classification head for BERT, we must know the number of output classes.

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
# 🔹 Load pre-trained BERT-base-uncased weights + add a Linear layer for classification.
# 🔸 Why BERT? Transformer architecture with self-attention that performs excellently in sentence understanding.
# 🔸 num_labels sets the classifier head output to [batch, num_labels].

model.to(device)
# 🔹 Move all model weights and buffers to the selected device (GPU or CPU).
# 🔸 On CPU, operations are slower but no GPU is required.

optimizer = AdamW(model.parameters(), lr=LR)
# 🔹 Define the AdamW optimizer for all model parameters with LR.
# 🔸 Why AdamW? Correct weight decay and better compatibility with transformers than vanilla Adam.

# -----------------------------
# 8. Training + evaluation
# -----------------------------
def evaluate(loader, name="Validation", silent=False):
    # 🔹 Evaluate the model on Val/Test without gradient computation (no_grad).
    # 🔸 Output: Accuracy and weighted F1 (suitable for imbalanced classes).
    model.eval()        # Evaluation mode: disable dropout/… for stable results
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            # 🔹 Read inputs from batch and move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["labels"].to(device)

            # 🔹 Forward pass without labels → produce logits
            logits = model(input_ids, attention_mask=attention_mask).logits

            # 🔹 Predicted label = index of the highest logit in each row
            preds.extend(torch.argmax(logits, -1).cpu().numpy())

            # 🔹 Collect gold (true) labels for comparison
            golds.extend(labels_b.cpu().numpy())

    # 🔹 Compute metrics on the full set
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="weighted")  # 🔸 weighted F1 is better for class imbalance

    if not silent:
        print(f"{name} Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


def train_model(epochs=EPOCHS):
    # 🔹 Training loop with scheduler, grad clipping, and saving the best model by Val F1
    total_steps = len(train_loader) * epochs
    # 🔹 Number of updates = #train batches per epoch × #epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    # 🔹 LR schedule: first 10% warmup (gradual increase), then linear decay to the end.
    # 🔸 Reason for warmup: early in training, weights are sensitive; smaller LR helps stability.
    # 🔸 Linear decay: slow down steps to stabilize toward the end of training.

    best_f1 = -1.0  # 🔹 Track best validation F1 to save the best weights

    for epoch in range(1, epochs + 1):
        model.train()      # 🔹 Training mode (dropout active, …)
        running_loss = 0.0 # 🔹 For averaging loss per epoch

        for batch in train_loader:
            optimizer.zero_grad()  # 🔹 Clear previous gradients

            # 🔹 Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["labels"].to(device)

            # 🔹 Forward with labels → the model computes CrossEntropyLoss internally.
            out = model(input_ids, attention_mask=attention_mask, labels=labels_b)
            loss = out.loss

            loss.backward()  # 🔹 Backpropagation: compute gradients for all parameters

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 🔹 Grad clipping: limit gradient norm to 1.0 to prevent exploding gradients.

            optimizer.step()  # 🔹 Update weights according to gradients and current LR
            scheduler.step()  # 🔹 Update LR according to schedule (warmup/decay)

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        # 🔹 Average loss for reporting training progress

        # 🔹 Evaluate on Val (no per-batch prints, just the final output)
        val_acc, val_f1 = evaluate(val_loader, name=f"Validation (epoch {epoch})", silent=True)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        if val_f1 > best_f1:
            # 🔹 If the model improved, save the weights to load later (Early Best)
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_combined.pt")

    print(f"Training done. Best val F1 = {best_f1:.3f}")



"""Why is F1 the criterion for selecting the best model?
Because the data may be imbalanced; the weighted F1 balances sensitivity/specificity.

Why Grad Clipping?
To control exploding gradients, which can occur during fine-tuning with a sensitive LR.

Warmup + Linear Decay?
Transformer training settings are usually more stable with warmup, and gradual decay helps smoother convergence.
"""



# -----------------------------
# 9. Predict & Act
# -----------------------------
custom_action_map = {
    label2id["open_browser_custom"]: open_browser,
    label2id["open_notepad_custom"]: open_notepad,
    label2id["open_calculator_custom"]: open_calculator,
    label2id["play_youtube_custom"]: play_youtube,
    label2id["greet_custom"]: say_hello,
    label2id["goodbye_custom"]: say_goodbye
}
# 🔹 Map “custom classes” to “real system action functions”.
# 🔸 Why only customs? Because for generic SNIPS classes no specific actions are defined (could be added later).

def predict_and_act(text, threshold=CONFIDENCE_THRESHOLD):
    # 🔹 This function takes user text → tries simple rules first (robust to ASR errors).
    # 🔸 If no rule triggers, it calls the BERT model and checks confidence with softmax.

    text_lower = text.lower()
    # 🔹 Lowercase for straightforward keyword matching (reduce sensitivity to case)

    # --- Fast and safe (Rule-based) rules ---
    if any(k in text_lower for k in ["browser", "open google", "open the browser", "open browser"]):
        open_browser(); return

    if "youtube" in text_lower:
        play_youtube(); return

    if any(k in text_lower for k in ["notepad", "text editor", "open notepad", "open text editor"]):
        open_notepad(); return

    if "calculator" in text_lower:
        open_calculator(); return

    if "time" in text_lower and "what" in text_lower or text_lower.strip().startswith("time"):
        # ⚠️ Subtle point: operator precedence (and before or) — this condition effectively means:
        #   ( both "time" and "what" are in the text ) or (the sentence starts with "time")
        # If you want it more explicit, you can add parentheses (but don’t change the current logic).
        tell_time(); return

    if any(g in text_lower for g in ["hello", "hi", "hey"]):
        say_hello(); return

    if any(b in text_lower for b in ["bye", "goodbye", "see you", "exit", "stop"]):
        say_goodbye(); return

    # --- If no simple rule matched → use the model (ML fallback) ---
    model.eval()
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # 🔹 Probability for each class
        confidence, pred = torch.max(probs, dim=-1)                  # 🔹 Max probability and corresponding class
        pred_id = int(pred.item())
        confidence = float(confidence.item())

    if confidence < threshold:
        # 🔹 If confidence is low, don’t execute—ask the user to repeat (avoid wrong actions)
        speak("Sorry, I didn't catch that clearly. Could you repeat?")
        return

    if pred_id in custom_action_map:
        # 🔹 If the predicted class is one of our custom ones → execute the corresponding action
        custom_action_map[pred_id]()
        return

    # 🔹 For other classes (original SNIPS), for now just report the intent name
    intent_name = id2label[pred_id] if pred_id in id2label else f"intent_{pred_id}"
    speak(f"I understood intent: {intent_name} (confidence {confidence:.2f}).")



"""
Why Rule-first?
ASR errors (like “open brow…”) can truncate text; simple keyword rules are robust.

Why a Confidence Threshold?
To avoid executing the wrong action; if confidence is low, we ask the user to repeat.

Possible improvements:
Add canonicalization (remove stopwords, correct misspellings, phrase normalization), increase custom phrases, or few-shot augmentation.
"""


# -----------------------------
# 10. Voice listening loop
# -----------------------------
def listen_loop():
    # 🔹 Main voice interaction loop: microphone → ASR (Google) → text → predict_and_act
    try:
        mic = sr.Microphone()  # 🔹 Use the system’s default audio input
    except Exception as e:
        print("Microphone error:", e)
        speak("Microphone not available. Exiting.")
        return

    r = sr.Recognizer()
    # 🔹 Initial sampling of ambient noise to set sensitivity (reduce false starts)
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)

    speak("Assistant is ready and listening. Say 'goodbye' to stop.")

    while True:
        r = sr.Recognizer()  # 🔹 New recognizer each turn (more isolation)
        with mic as source:
            print("Listening... (speak now)")
            try:
                # 🔹 timeout: max time to wait for start of speech
                # 🔹 phrase_time_limit: max duration of each turn
                audio = r.listen(source, timeout=12, phrase_time_limit=7)

                # 🔹 Convert speech to text with Google’s service (requires internet)
                text = r.recognize_google(audio)
                print("You said:", text)

            except sr.WaitTimeoutError:
                # 🔹 If the user didn’t speak (long silence), continue the loop (no error)
                continue
            except sr.UnknownValueError:
                # 🔹 If ASR couldn’t understand (noise/accent/unusual grammar)
                speak("Sorry, I couldn't understand. Please repeat.")
                continue
            except sr.RequestError as e:
                # 🔹 Network error/Google service unavailable
                print("Speech recognition request error:", e)
                speak("There is a problem with the speech recognition service.")
                time.sleep(2)
                continue

        # 🔹 Safe exit phrases: user can close the program by saying stop/bye/...
        if text.lower().strip() in {"stop", "quit", "exit", "goodbye", "bye"}:
            say_goodbye()
            break

        # 🔹 Otherwise, interpret the text and execute the corresponding action
        predict_and_act(text)

"""

Why adjust_for_ambient_noise?
To adapt the recognizer’s internal threshold to ambient noise and reduce false positives.

timeout and phrase_time_limit?
UX control: don’t wait too long, and don’t allow overly long monologues.

UnknownValueError vs RequestError?
The former is when the audio isn’t understood, the latter is when the service/internet is down.
"""


# -----------------------------
# 11. Main
# -----------------------------
if __name__ == "__main__":
    # 🔹 Program entry point: when the file is run directly (not imported as a module)

    if os.path.exists("best_combined.pt"):
        # 🔹 If the best model was previously saved, load it (save time and ensure reproducibility)
        print("Loading saved model 'best_combined.pt' ...")
        model.load_state_dict(torch.load("best_combined.pt", map_location=device))
    else:
        # 🔹 Otherwise, run training
        print("Training model on SNIPS + custom intents ...")
        train_model(epochs=EPOCHS)
        # 🔹 After training, if the best model file was created, load it (so inference uses the exact best)
        if os.path.exists("best_combined.pt"):
            model.load_state_dict(torch.load("best_combined.pt", map_location=device))

    # 🔹 Final evaluation on Test: only once (to avoid leaking into Test during development)
    evaluate(test_loader, name="Test")

    # 🔹 Start interaction: a short greeting and then enter the listening loop
    say_hello()
    listen_loop()



"""
Why Test only once?
To avoid “unintentional manual tuning” based on Test results (leakage/overfitting to Test).

Why reload after training?
Because during training we saved the best weights by F1; we thus ensure inference uses the exact best model.
"""
