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

Here are some results of the execution on an AMD R7 260X GPU with 14.12 AMD Catalyst driver on a 64bit Linux system:

```
1. Shared memory only kernel
Time: 0.089481 msecs (0.732401 billion elements/second)
2. Hybrid kernel via subgroup functions
Time: 0.215851 msecs (0.303617 billion elements/second)
3. Workgroup function kernel
Time: 0.475408 msecs (0.137852 billion elements/second)
```

