# load-data-signai

Demonstration repository showing how to load pose/hand/face **landmark data** for
**Colombian Sign Language** (*LSC — Lengua de Señas Colombiana*) video classification.

Instead of raw video, this dataset stores pre-extracted body landmarks (one array per
clip). It is meant as a minimal, reproducible example of how the landmark data is
organized and how to load it for training a sign classifier.

## Signs (classes)

The dataset currently covers 5 signs from LSC. Each sign is a folder under
`data/landmarks/` named after its class number:

| Class | Folder | Sign (Spanish) | Meaning (English) |
|------:|--------|-----------------|--------------------|
| 1 | [data/landmarks/sign_01](data/landmarks/sign_01) | Café | Coffee |
| 2 | [data/landmarks/sign_02](data/landmarks/sign_02) | Aromática | Herbal/aromatic tea |
| 3 | [data/landmarks/sign_03](data/landmarks/sign_03) | Hola | Hello |
| 4 | [data/landmarks/sign_04](data/landmarks/sign_04) | ¿Qué necesitas? | What do you need? |
| 5 | [data/landmarks/sign_05](data/landmarks/sign_05) | Con gusto | You're welcome / With pleasure |

These signs were chosen because they represent a small, realistic interaction in a
coffee-shop setting — greeting a customer, asking what they need, and naming the two
most common orders (coffee and herbal tea), then closing the interaction politely.

## Data layout

```
data/
└── landmarks/
    ├── sign_01/           # Clips for "Café"
    ├── sign_02/           # Clips for "Aromática"
    ├── sign_03/           # Clips for "Hola"
    ├── sign_04/           # Clips for "¿Qué necesitas?"
    └── sign_05/           # Clips for "Con gusto"
```

Each `sign_0N` folder holds exactly 300 `.npy` files, for 1,500 clips total. There is
no separate index file — the class label for a clip comes directly from the
`sign_0N` folder it lives in.

Each sign folder contains one `.npy` file per **clip**. A clip is a short recording of
a single performance of the sign, already segmented into frames. Each `.npy` file has
shape `(num_frames, 1662)`:

- `num_frames` varies per clip (different clips have different lengths — e.g. 25, 52,
  67, 71 frames, etc.), since each performance of the sign takes a different amount of
  time.
- `1662` is the number of landmark values extracted per frame, matching the standard
  MediaPipe Holistic layout: pose (33 points × 4 values), face mesh (468 points × 3
  values), left hand (21 points × 3 values), and right hand (21 points × 3 values) —
  `33*4 + 468*3 + 21*3 + 21*3 = 1662`.

So a single `.npy` file is effectively "all the landmarks for every frame of one
clip".

## Filename convention

Every clip file follows this pattern:

```
S<sign>_M<participant>_V<view>_C<camera>_Clip<clip_id>.npy
```

For example, `S01_M01_V01_C01_Clip001.npy` means:

- `S01` — **Sign 01** ("Café")
- `M01` — **Participant 01**, one of the four signers who recorded the corpus
- `V01` — **View/capture angle 01**
- `C01` — **Camera 01** used to record it
- `Clip001` — Clip 1, i.e. this specific segmented recording

`S`, `M`, `V`, and `C` are zero-padded to 2 digits; `Clip` is zero-padded to at least
3 digits (4 digits once the global clip id passes 999). Clip IDs are numbered globally
and continuously across the whole dataset (sign 1 uses clips 1–300, sign 2 uses
301–600, and so on), not restarted per sign — but the loader does not rely on that,
since the class label is always read from the `sign_0N` folder a file lives in.

Each sign folder has 300 clips (4 participants × 3 views × 25 repetitions), for 1,500
clips total across the 5 signs.

## Loading the data

The `signai_landmarks` package (in [src/signai_landmarks](src/signai_landmarks))
implements this loading logic:

```python
from signai_landmarks import load_dataset

X, y = load_dataset("data/landmarks")
# X: object-dtype ndarray, shape (1500,), each element a (num_frames, 1662) array
# y: int ndarray, shape (1500,), the sign class (1-5) for each clip
```

Install it locally with `pip install -e .` (or `pip install -e ".[dev,notebook]"` to
also get the test and notebook dependencies).

Since `num_frames` differs per clip, sequences must be padded/truncated to a fixed
length (or fed to a model that supports variable-length input) before batching, e.g.
with `tf.keras.preprocessing.sequence.pad_sequences` or an equivalent PyTorch
padding utility.

See [notebooks/explore_dataset.ipynb](notebooks/explore_dataset.ipynb) for a worked
example that loads the dataset and visualizes class balance, sequence lengths, and
the extracted pose/hand skeleton.

## License

CC BY 4.0 — see [LICENSE](LICENSE).
