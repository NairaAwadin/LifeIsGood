Goal
- Turn a listing into a decision:
  * 0 = "no contradiction found (or empty/no evidence)"
  * 1 = "description contradicts city/postalCode"
- The model is a function: text + context -> probability of contradiction.

Big picture pipeline (inference)

1) Define what information you give the model
- What: Decide the input fields (postalCode, city, description) and how to combine them.
- Why: Models don't "see columns" - they see tokens. You must present the row as one text so the model can compare structured fields with description.
- Typical format:
  [POSTAL] 75014
  [CITY] Paris 14e
  [DESC]
  ...
  (Tags act like column headers for the model.)

2) Tokenize (convert text -> numbers)
- What: Use the tokenizer to produce input_ids (token numbers) and attention_mask (real vs padding).
- Why: Neural models process token IDs, not raw text.
- Key: Tokenizer must match the model used in training.

3) Load the model weights (the learned brain)
- What: Load saved model parameters + configuration.
- Why: The knowledge is in the weights; without them you have a random model.
- Conceptually: config = architecture + label mapping; weights = learned patterns.

4) Set inference mode and choose hardware
- What: call eval(); choose CPU vs GPU; move model and inputs to that device.
- Why: eval() for stable predictions; device alignment for speed and correctness.

5) Forward pass: get scores
- What: Run the model on tokenized input -> logits (scores for class 0 and 1).
- Why: Logits are the model's raw judgment before probabilities.

6) Convert to probability and decide a threshold
- What: Apply softmax -> P(label=1); compare to a threshold (default 0.5).
- Why: Probability gives confidence; threshold sets precision/recall tradeoff.
- For higher precision (fewer false positives), consider threshold 0.7-0.9.

Training pipeline (why we needed a dataset)
- Train on many examples in the same input format; adjust weights so:
  * footer addresses -> label 0
  * explicit contradictions -> label 1
  * implicit contradictions ("15 km de Paris") -> label 1 (if labeled so)
  * empty description -> label 0
- Model learns patterns statistically from labeled examples.

Why we don't just send description alone
- Contradiction depends on what it should match. The structured fields must be in the input so the model can compare them to the description.

One design decision to make now
- Choose output type:
  * binary (0/1 only), or
  * binary + confidence (recommended), or
  * 3 classes (contradiction / consistent / insufficient evidence).
- Once you choose, set the inference steps accordingly (thresholds, handling empty descriptions, etc.).
