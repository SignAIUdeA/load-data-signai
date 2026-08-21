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
| 1 | [data/landmarks/1](data/landmarks/1) | Café | Coffee |
| 2 | [data/landmarks/2](data/landmarks/2) | Aromática | Herbal/aromatic tea |
| 3 | [data/landmarks/3](data/landmarks/3) | Hola | Hello |
| 4 | [data/landmarks/4](data/landmarks/4) | ¿Qué necesitas? | What do you need? |
| 5 | [data/landmarks/5](data/landmarks/5) | Con gusto | You're welcome / With pleasure |

These signs were chosen because they represent a small, realistic interaction in a
coffee-shop setting — greeting a customer, asking what they need, and naming the two
most common orders (coffee and herbal tea), then closing the interaction politely.

## Data layout

```
data/
└── landmarks/
    ├── data_curated.csv   # File -> Class index for every clip
    ├── 1/                 # Clips for "Café"
    ├── 2/                 # Clips for "Aromática"
    ├── 3/                 # Clips for "Hola"
    ├── 4/                 # Clips for "¿Qué necesitas?"
    └── 5/                 # Clips for "Con gusto"
```

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

Every clip file (and its row in `data_curated.csv`) follows this pattern:

```
S<sign>V<video>C<camera>M<model><variant> Clip<clip_id>.npy
```

For example, `S1V1C1M1A Clip1.npy` means:

- `S1` — **Seña 1** (Sign 1, "Café")
- `V1` — **Video/take 1** for that sign
- `C1` — **Camera/angle 1** used to record it
- `M1A` — **Modelo 1**, variant `A` (the signer/session performing the sign; the
  variant letter groups recordings from the same performer/session and is shared by
  every clip of that sign)
- `Clip1` — Clip 1, i.e. this specific segmented recording

Each sign has its own combination of video takes, camera angles and models, and
therefore a different number of clips per sign — they are **not** interchangeable
across signs. For this dataset, every sign has 4 video takes × 3 cameras × 25
repetitions = 300 clips, for 1500 clips in total across the 5 signs. Clip IDs are
numbered globally and continuously across the whole dataset (e.g. sign 1 uses clips
1–300, sign 2 uses 301–600, and so on), rather than restarting per sign.

## `data_curated.csv`

`data/landmarks/data_curated.csv` is the index that ties every clip file to its class
label:

| Column | Description |
|--------|--------------|
| `File` | Clip filename (matches a `.npy` file inside its class folder) |
| `Class` | Sign/class number (`1`–`5`), matching the folder the file lives in |

```csv
File,Class
S1V3C1M3A Clip166.npy,1
S1V1C1M1A Clip18.npy,1
S4V1C2M1D Clip929.npy,4
S5V1C1M1D Clip1202.npy,5
```

## Loading the data

```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/landmarks")
index = pd.read_csv(DATA_DIR / "data_curated.csv")

sequences, labels = [], []
for _, row in index.iterrows():
    landmarks = np.load(DATA_DIR / str(row["Class"]) / row["File"])  # (frames, 1662)
    sequences.append(landmarks)
    labels.append(row["Class"])
```

Since `num_frames` differs per clip, sequences must be padded/truncated to a fixed
length (or fed to a model that supports variable-length input) before batching, e.g.
with `tf.keras.preprocessing.sequence.pad_sequences` or an equivalent PyTorch
padding utility.

## License

MIT — see [LICENSE](LICENSE).
