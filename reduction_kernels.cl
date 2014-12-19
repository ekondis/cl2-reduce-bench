#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 1  
#endif

__kernel void reductionShmem(__local volatile uint *localBuffer, __global uint *result/*, const unsigned int n*/) {
	const uint id = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint group_size = get_local_size(0);

	// initialize shared memory contents
	uint res = id;            // reduce 0+1+2+3+...+max_global_id
	localBuffer[lid] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	// local memory reduction
	int i = group_size/2;
	for(; i>WAVEFRONT_SIZE; i >>= 1) {
		if(lid < i)
			localBuffer[lid] = res = res + localBuffer[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// wavefront reduction
	for(; i>0; i >>= 1) {
		if(lid < i)
			localBuffer[lid] = res = res + localBuffer[lid + i];
	}
	// atomic reduce in global memory
	if(lid==0)
		atomic_add(result, res);
}  

#ifdef K2

__kernel void reductionSubgrp(__local volatile uint *localBuffer, __global uint *result) {
	const uint id = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint group_size = get_local_size(0);

	// initialize shared memory contents
	uint res = id;            // reduce 0+1+2+3+...+max_global_id
	localBuffer[lid] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	// local memory reduction
	int i = group_size/2;
	for(; i>WAVEFRONT_SIZE; i >>= 1) {
		if(lid < i)
			localBuffer[lid] = res = res + localBuffer[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// subgroup reduction (introduced in OpenCL 2.0)
	if(lid < i)
		res = sub_group_reduce_add(res + localBuffer[lid + i]);

	// atomic reduce in global memory
	if(lid==0)
		atomic_add(result, res);
}

#endif

#ifdef K3

__kernel void reductionWkgrp(__global uint *result) {
	const uint id = get_global_id(0);
	const uint lid = get_local_id(0);

	uint res = id;            // reduce 0+1+2+3+...+max_global_id

	// workgroup reduction (introduced in OpenCL 2.0)
	res = work_group_reduce_add(res);

	// atomic reduce in global memory
	if(lid==0)
		atomic_add(result, res);
}

#endif
