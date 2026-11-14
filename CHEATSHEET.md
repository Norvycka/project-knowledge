# Math Cheatsheet: Phase 1

Plain-words review of Lessons 1.01 to 1.04, plus the start of 1.05. Every example is one we worked through together.

---

## Words to know

| Word | Plain meaning | Your example |
|---|---|---|
| **Vector** | A list of numbers that describes something. Also an arrow. | `rengar = [9.9, 4.0, 0.0, 0.0, 0.0]` |
| **Embedding** | A vector that captures the *meaning* of something (word, image, user, champion) | Your `rengar` vector is a hand-made embedding |
| **Matrix** | A stack of rows. Each row is a vector. | `champions` = 3 rows × 5 columns |
| **Shape** | `rows × columns`. It's a label, nothing gets multiplied. | `champions` is `(3, 5)` |
| **Dot product** | Multiply matching numbers, then add them all up → one number | `rengar · talon = 110.01` |
| **Element-wise** (`*`) | Multiply matching positions, but **don't** add up | `[2, 3] * [4, 5] = [8, 15]` |
| **Matrix multiply** (`@`) | One dot product per row | `champions @ me = [114.01, 110.01, 0.0]` |
| **Weight** (`w`, `W`) | The knobs a model learns. A matrix of them is `W`. | `W = np.random.randn(2, 3)` |
| **Bias** (`b`) | A fixed amount added on at the end, the same for everyone | The patch bonus `[-10, +5, +20]` |
| **Broadcasting** | NumPy copies a smaller array to fit a bigger one | The bias row copied onto every player |
| **Transpose** (`.T`) | Swap rows and columns | `(3, 5)` becomes `(5, 3)` |
| **Identity matrix** | 1s on the diagonal, 0s elsewhere. The "do nothing" matrix. | `np.eye(3) @ [7, 8, 9] = [7, 8, 9]` |
| **Rank** | How many columns bring genuinely *new* information | 6 columns, one is a sum of others → rank 5 |
| **Determinant** | One number: how much a matrix scales area. 0 = squashed flat. | `det([[2,0],[0,3]]) = 6` |
| **Eigenvector** | A direction the matrix only stretches, never turns | `[1, 1]` → `[3, 3]` |
| **Eigenvalue** | How much the eigenvector gets stretched | `3` |
| **Derivative** | One slope: "nudge the input, how much does the output change?" | `(4.41 − 4) / 0.1 = 4.1` |
| **Gradient** | All the slopes together, one per knob, as a vector | `[dw, db] = [4, 2]` |
| **Learning rate** (`lr`) | Step size when walking downhill | `0.01` |
| **Loss** | One number for "how wrong". The height on the hill. | `error² = 9` |

---

## 1.01: Vectors and the dot product

**A vector is a list of numbers with a fixed order.** Each position always means the same thing.

```text
          [assassin, bruiser, mage, marksman, healer]
rengar  = [   9.9,      4.0,   0.0,    0.0,     0.0 ]
talon   = [   9.9,      3.0,   0.0,    0.0,     0.0 ]
soraka  = [   0.0,      0.0,   6.0,    0.0,     9.0 ]
```

**List view vs arrow view:** `[3, 2]` means "walk 3 right, 2 up". Draw a line from the start to where you end up. That's the arrow. Same thing, two ways to look at it. 5 numbers = 5 directions (can't draw it, but the maths is the same).

**Dot product = multiply matching positions, then add up:**

```text
rengar · talon  = 9.9×9.9 + 4.0×3.0 + 0 + 0 + 0 = 98.01 + 12 = 110.01   ← very alike
rengar · soraka = 9.9×0   + 4.0×0   + 0×6 + 0×0 + 0×9          = 0        ← nothing in common
```

**Big dot product = the two point the same way = similar.** This is how search, recommendations and ChatGPT's memory search work.

Your code:

```python
def dot(a, b):
    total = 0
    for x, y in zip(a, b):      # walks both lists side by side, one position at a time
        total = total + x * y   # running total: old total + this pair's product
    return total
```

More numbers in the vectors = **more loop rounds**, not more variables.

---

## 1.01: Matrix × vector (your recommender)

**A matrix is a stack of vectors.** Matrix × vector = **one dot product per row.**

```text
champions (3 × 5)          me (5)                scores (3)
[rengar row]   · me   →   114.01
[talon row]    · me   →   110.01
[soraka row]   · me   →     0.0
```

```python
def matvec(M, v):
    results = []
    for row in M:
        results.append(dot(row, v))   # one score per row
    return results                    # don't forget return!
```

**This is one neural network layer.** The only difference: a real layer's numbers are *learned*, not typed in.

---

## 1.01: Independence and rank

**A column is dependent if you can build it from the others** (by multiplying columns by numbers and adding them).

```text
damage_dealer = assassin + mage + marksman     ← adds NOTHING new
height_m      = height_cm / 100                ← adds NOTHING new
```

**Rank = number of columns − number of columns you can build from the others.**

| Table | Columns | Dependent | Rank |
|---|---|---|---|
| assassin, mage, healer, total_damage | 4 | 1 | **3** |
| 3 columns, third = first + second | 3 | 1 | **2** |

**Trap:** "none of the columns are identical" does **not** mean independent. A column can be a *combination* without being a copy.

**LoRA** (fine-tuning big models cheaply): instead of updating a 4096 × 4096 weight matrix (16.7 million numbers), it trains two thin ones, 4096 × 16 and 16 × 4096 (~131,000 numbers). The bet: the *change* only needs a few independent directions (low rank). Too low → raise the rank.

---

## 1.02: Shapes

```text
(m × n) @ (n × p) = (m × p)
     inner numbers must MATCH and disappear
     outer numbers SURVIVE
```

**Plain rule: rows decide how many come out. Columns must match the input.**

| Matrix | Input | Output | Why |
|---|---|---|---|
| `champions` (3 × 5) | `me` (5) | 3 scores | one per champion (row) |
| 6 restaurants × 3 features | taste (3) | 6 scores | one per restaurant |
| `W` (128 × 784) | image (784 pixels) | 128 numbers | one per detector |
| `W` (2 × 3) | `x` (3) | 2 numbers | `W` shape = (outputs, inputs) |

Input with the wrong length (e.g. 4 instead of 5) → **shape mismatch error**.

---

## 1.02: `*` vs `@`

```text
rengar * me  →  [98.01, 16.0, 0.0, 0.0, 0.0]   element-wise: multiply only
rengar @ me  →  114.01                         matrix multiply: multiply AND add up
```

In a layer you almost always want `@`. Using `*` by mistake is a silent bug.

---

## 1.02: Bias and broadcasting

**Bias = a fixed amount added to each output, regardless of the input.** Like a patch: same for every player.

```text
champions @ me          = [114.01, 110.01,  0.0]
+ bias (patch)          = [ -10,     +5,    +20]
= scores                = [104.01, 115.01, 20.0]   ← Talon now on top
```

**Broadcasting:** the bias has 3 numbers, the table has 2 rows (2 players). NumPy **copies the bias onto every row**, then adds:

```text
[114.01, 110.01,  0.0]     [-10, +5, +20]     [104.01, 115.01,  20.0]   ← you
[ 20.0,   30.0,  90.0]  +  [-10, +5, +20]  =  [ 10.0,   35.0,  110.0]   ← friend
```

It's OK to copy because the patch is the same for everyone.

---

## 1.02: Detectors (what `W` really does)

Each **row of `W` is a pattern detector**. The dot product with an image says *how strongly that pattern shows up*.

```text
2×2 image, flattened:   ■ ■ / □ □  →  [1, 1, 0, 0]
detector "bright top":                [1, 1, 0, 0]
score = 1 + 1 + 0 + 0 = 2   ← full match
```

| Image | Score | Meaning |
|---|---|---|
| top row bright `[1,1,0,0]` | 2 | full match |
| left column bright `[1,0,1,0]` | 1 | half match |
| bottom row bright `[0,0,1,1]` | 0 | no match |

**Three kinds of numbers:** weight (inside `W`, like `9.9`), input (`me`, the image), output (the score, like `114.01`).

**Why detectors matter:** a shifted "3" has totally different pixels but the same patterns. Layers build up: edges → parts → whole digits.

---

## 1.02: Determinant and inverse

```text
| a  b |
| c  d |     det = a×d − b×c
```

| Matrix | det | Meaning |
|---|---|---|
| `[[1, 2], [3, 4]]` | 4 − 6 = **−2** | fine, can be undone |
| `[[1, 2], [2, 4]]` | 4 − 4 = **0** | row 2 = 2 × row 1 → **no inverse** |
| `[[9, 4], [18, 8]]` (Rengar + clone) | 72 − 72 = **0** | clone = Rengar doubled |

**det = 0 means a redundant row (the cause) → it squashes a dimension → it can't be undone** (`LinAlgError: Singular matrix`).

**Trap:** det = 0 does **not** mean the matrix is full of zeros. `[[1, 2], [2, 4]]` has no zeros at all.

---

## 1.02: The layer

```python
output = np.maximum(0, W @ x + b)
```

| Piece | Recommender version |
|---|---|
| `x` | your taste (input) |
| `W` | champion table, but random numbers |
| `W @ x` | one score per row |
| `+ b` | the patch bonus |
| `np.maximum(0, ...)` = **ReLU** | negative scores become 0 ("detector stays silent") |

Write it with **variable names**, never the numbers typed in.

---

## 1.03: Matrices as machines that move points

Read each **row** as a recipe for one new coordinate:

```text
| 2  0 |   new x = 2·x + 0·y      the 0s mean "no mixing"
| 0  3 |   new y = 0·x + 3·y
```

| Matrix | What it does | det |
|---|---|---|
| `[[2, 0], [0, 3]]` | **stretch** x ×2, y ×3 (square → 2 × 3 rectangle) | 6 (area ×6) |
| `[[2, 1], [0, 3]]` | **shear**: leans (height pushes points sideways) | 6 |
| `[[-1, 0], [0, 1]]` | **mirror** across the vertical axis | −1 (flipped) |
| `[[0, -1], [1, 0]]` | **rotate** 90° counter-clockwise | 1 |
| `[[1, 0], [0, 1]]` | identity, does nothing | 1 |

**(0, 0) never moves** under any matrix (anything × 0 = 0). That's why layers need the bias: `W @ x + b` can slide things, `W @ x` can't.

**Test with lopsided shapes.** A square (or an equal-arm L) can make a rotation and a mirror look identical.

---

## 1.03: Order matters

Read chains **right to left**: in `S2 @ R`, `R` happens first.

```text
rotate first, then stretch x ×2   → long bar lies flat, then doubles  → length 4
stretch first, then rotate        → long bar squashed first, then lies flat → length 1
```

**`S2 @ R ≠ R @ S2`.** Swapping the order changes the result ("not commutative").

---

## 1.03: Eigenvectors

Most arrows get **turned** by a matrix. Eigenvectors only get **stretched**.

```text
A = [[2, 1], [1, 2]]

A @ [1, 0]  = [2, 1]     turned        → not an eigenvector
A @ [1, 1]  = [3, 3]     same direction, ×3  → eigenvector, eigenvalue 3
A @ [1, -1] = [1, -1]    unchanged     → eigenvector, eigenvalue 1
```

Each row → **one** number (multiply AND add). Not 6, not `[2, 0]`.

**Repeating a matrix** (like an RNN does) multiplies by the eigenvalue every step:

| Eigenvalue | After 10 steps | Name |
|---|---|---|
| 3 | 3¹⁰ = **59,049** | exploding |
| 1 | stays the same | stable |
| 0.5 | 0.5¹⁰ ≈ 0.001 | vanishing |

It multiplies, it doesn't add (not 60). The eigenvalue isn't written in the code: it's a property of the matrix, discovered by testing.

---

## 1.04: The hill in fog

- **Hill height** = loss (how wrong)
- **Your position** = a weight's value
- **Slope under your feet** = derivative

```text
slope ≈ (error after a tiny step − error now) / step size
      = (2.1² − 2²) / 0.1 = (4.41 − 4) / 0.1 = 4.1
```

| Slope | Meaning | Move w |
|---|---|---|
| positive | uphill to the right | smaller |
| negative | uphill to the left | bigger |
| 0 | at the bottom | stop |

**Gradient descent:** `w = w − lr × slope`. **Always minus.** The slope's sign picks the direction for you. (Using `+` climbs the hill: gradient *ascent*, your bug.)

**Learning rate:** too big (1.1) jumps over the valley and explodes. Too small crawls. Near the bottom you creep closer forever (0.0005), never exactly 0.

---

## 1.04: The 5-step training card

One data point: `x = 2`, true `y = 5`, `lr = 0.01`.

```text
                 Run 1: w=1, b=0        Run 2: w=3, b=0
1. GUESS         pred  = w×x + b        2                      6
2. HOW WRONG     error = pred − y       −3  (too low)          +1  (too high)
3. ONE SCORE     loss  = error²         9                      1
4. FEEL SLOPE    dw    = 2 × error × x  −12 (negative → up)    +4  (positive → down)
                 db    = 2 × error      −6                     +2
5. STEP          w     = w − lr × dw    1.12                   2.96
                 b     = b − lr × db    0.06                   −0.02

check:           new guess               ~2.30 (closer to 5)    5.90 (closer to 5)
```

**One sentence:** the guess was too high → the slope was positive → the minus rule made w go down.

**The real `y` is the answer key.** Only the starting `w` and `b` are random. We train on known answers so the model can predict **new** `x` values where nobody knows `y` (like predicting a new League game from 10,000 past ones).

**`train.py` = this card**, except: all 5 points at once (`np.mean` averages them), both `w` and `b`, repeated 1000 times (`for epoch in range(1000)`). Different random starts all end near `w = 2`, `b = 1` (the rule `y = 2x + 1`).

**Don't memorise slope formulas.** PyTorch computes them for you (`loss.backward()`).

---

## 1.05 (in progress): The chain rule

**Gears:** A turns B 3 times, B turns C 2 times → A turns C **3 × 2 = 6** times.
**When things are linked in a chain, their rates multiply.**

Your training step is a chain:

```text
w   →   pred   →   error   →   loss
   (×x)       (−y)        (squared)
```

Rates for `w = 3`, `x = 2`, `y = 5` (so `pred = 6`, `error = 1`):

| Link | Rate | How we found it |
|---|---|---|
| `w` → `pred` | **x = 2** | w 3 → 4 made pred 6 → 8 |
| `pred` → `error` | **1** | pred 6 → 7 made error 1 → 2 |
| `error` → `loss` | **2 × error = 2** | slope of a square is 2 × the thing |

**Pick up here:** multiply the three rates. Compare with the `dw` from Run 2 above.

---

## Your review list

- **Rank** = columns minus the ones you can build from others (not just "count the columns")
- **det = 0** means a redundant row, not zeros (and no inverse)
- **Port forwarding**: `ssh -L 8888:localhost:8888 user@box`
- **Vocabulary**: identity matrix, element-wise, embedding, derivative, gradient
