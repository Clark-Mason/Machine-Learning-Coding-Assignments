
TREE
+-- [SPLIT: x4 = 1]
|       +-- [LABEL = 1]
+-- [SPLIT: x4 = 1]
|       +-- [SPLIT: x0 = 1]
|       |       +-- [SPLIT: x1 = 1]
|       |       |       +-- [LABEL = 1]
|       |       +-- [SPLIT: x1 = 1]
|       |       |       +-- [LABEL = 0]
|       +-- [SPLIT: x0 = 1]
|       |       +-- [SPLIT: x1 = 1]
|       |       |       +-- [LABEL = 1]
Test Error = 33.33%.

Splits off same feature twice? somewhere i am not ensuring the same feature value cannot be used twice. Probably in the id3 or partition function
