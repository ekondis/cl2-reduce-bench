cl2-reduce-bench
================

This is a test case program for OpenCL 2.0 devices written in order to test the performance of workgroup and subgroup reduction functions introduced in OpenCL 2.0 API. 


Reduction operation
--------------

The problem applied is to find the sum 1+2+3+...+N which is an artificial problem with an easy to verify result. Each workitem is assigned a term of the expression. Thereafter, the reduction is performed in one or two stages. This is expressed in 3 different kernels:

1. Shared memory only kernel
2. Hybrid kernel via subgroup functions
3. Workgroup function kernel

The first is an OpenCL 1.1 implementation which uses shared memory to perform the intra-workgroup reduction. The second uses shared memory for the inter-wavefront reductions of the same workgroup and the subgroup reduce function (sub_group_reduce_add) for the intra-wavefront reduction. The last version uses the reduce workgroup function (work_group_reduce_add) and not shared memory at all. All implementations use global memory atomic to perform the reduction of values between different workgroups. As long only one atomic operation is required per each workgroup the performance degradation due to serialization is kept to minimum.

Example execution
---------------

Here are some results of the execution on an AMD A6-1450 APU, using 15.9 AMD Catalyst driver on a 64bit Ubuntu Linux system:

```
1. Shared memory only kernel
Executing...Done!
Output: 4294901760 / Time: 0.13121 msecs (0.998947 billion elements/second)
PASSED!

2. Hybrid kernel via subgroup functions
Executing...Done!
Output: 4294901760 / Time: 0.370266 msecs (0.353994 billion elements/second)
Relative speed-up with respect to kernel 1: 0.354367 (2.82193 times slower)
PASSED!

3. Workgroup function kernel
Executing...Done!
Output: 4294901760 / Time: 1.44436 msecs (0.0907475 billion elements/second)
Relative speed-up with respect to kernel 1: 0.0908431 (11.008 times slower)
PASSED!
```
