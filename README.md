In this paper, we propose the two-stage density clustering algorithm, which takes advantage of granular computing to address the aforementioned issues.

The new algorithm is highly efficient, adaptive to various types of data, and requires minimal parameter setting.
The first stage uses the two-round-means algorithm to obtain $\sqrt{n}$ small blocks, where $n$ is the number of instances.
This stage decreases the data size directly from $n$ to $\sqrt{n}$.
The second stage constructs the master tree and obtains the final blocks.
This stage borrows the structure of CFDP, while the cutoff distance parameter is not required.

The time complexity of the algorithm is $O(mn^\frac{3}{2})$, which is lower than $O (mn^2)$ for CFDP.
