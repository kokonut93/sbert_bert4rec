'''this pyfile generates the similarity matrix based on the below information
1. meta-information of items in Region A (with size n)
2. meta-information of items in Region B (with size m)

Suppose that Region B is Target Region, the aim is to create Similarity Score Matrix
between Region A and B with the size of n X m.
By doing so, we can create matrix where for each item in Region A,
similarity scores for all items across Region B are calculated.


Few dependencies exist.
1. The number of items in both Region A, and B should be considered.
'''