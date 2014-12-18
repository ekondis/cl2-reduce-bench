#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>  
#include <CL/cl.hpp>

inline std::string trim(std::string str) {
	str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
	str.erase(str.find_last_not_of(' ')+1);         //surfixing spaces
	return str; // use substring...
}

int main(void) {  
	VECTOR_CLASS<cl::Platform> platforms;
	VECTOR_CLASS<cl::Device> devices;
	std::cout << "OpenCL platform/device selection" << std::endl;
	cl::Platform::get(&platforms);
	std::cout << "Total platforms: " << platforms.size() << std::endl;
	std::string tmp;
	int iP = 1, iD;
	for(VECTOR_CLASS<cl::Platform>::iterator pl = platforms.begin(); pl != platforms.end(); ++pl) {
		if( platforms.size()>1 )
			std::cout << iP++ << ". ";
		std::cout << pl->getInfo<CL_PLATFORM_NAME>() << std::endl;
		pl->getDevices(CL_DEVICE_TYPE_ALL, &devices);
		iD = 1;
		for(VECTOR_CLASS<cl::Device>::iterator dev = devices.begin(); dev != devices.end(); ++dev)
			std::cout << '\t' << iD++ << ". " << trim(dev->getInfo<CL_DEVICE_NAME>()) << '/' << dev->getInfo<CL_DEVICE_VENDOR>() << std::endl;

	}
	if( platforms.size()>1 ){
		std::cout << "Select platform index: ";
		std::cin >> iP;
		while( iP<0 || iP>(int)platforms.size() ){
			std::cout << "Invalid platform index. Select again:";
			std::cin >> iP;
		}
	} else
		iP = 1;
	cl::Platform pl = platforms[iP-1];
	pl.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	
	if( devices.size()>1 ){
		std::cout << "Select device index: ";
		std::cin >> iD;
		while( iD<0 || iD>(int)devices.size() ){
			std::cout << "Invalid device index. Select again:";
			std::cin >> iD;
		}
	} else
		iD = 1;

	cl::Device dev = devices[iD-1];

	cl_device_type devType;
	dev.getInfo(CL_DEVICE_TYPE, &devType);
//	std::cout << "Type: " << devType << std::endl;

	int wavefront_size=1;
	if( devType==CL_DEVICE_TYPE_GPU ){
		pl.getInfo(CL_PLATFORM_NAME, &tmp);
		if( tmp.find("AMD") != std::string::npos )
			wavefront_size = 64;
		if( tmp.find("NVIDIA") != std::string::npos )
			wavefront_size = 32;
	}

	dev.getInfo(CL_DEVICE_NAME, &tmp);  
	std::cout << "Selected device:   " << tmp << std::endl;

	dev.getInfo(CL_DEVICE_VERSION, &tmp);
//	std::cout << "OpenCL version:   " << tmp << std::endl;

//	dev.getInfo(CL_DRIVER_VERSION, &tmp);
//	std::cout << "OpenCL version:   " << tmp << std::endl;

//    std::string deviceVerStr(deviceVersion);
    size_t vStart = tmp.find(' ', 0), vEnd = tmp.find('.', 0);
	int CLMajorVer = std::stoi( tmp.substr(vStart + 1, vEnd - vStart - 1) );
	std::string cl_version_option;
	bool cl_subgroups = false, cl_ver20 = false;
	if( CLMajorVer>=2 ) {
		std::cout << "Great! OpenCL 2.0 is supported :)" << std::endl;
		cl_ver20 = true;
		cl_version_option = "-cl-std=CL2.0 -cl-uniform-work-group-size -DK3";
		dev.getInfo(CL_DEVICE_EXTENSIONS, &tmp);
		std::string extension;
		std::stringstream sstmp(tmp);
		while( sstmp >> extension )
			if( extension=="cl_khr_subgroups" )
				cl_subgroups = true;
		if( cl_subgroups )
			cl_version_option += " -DK2";
	} else {
		std::cout << "Using OpenCL 1.X" << std::endl;
		cl_version_option = "-cl-std=CL1.1";
	}
	if( wavefront_size>1 )
		cl_version_option += " -DWAVEFRONT_SIZE="+std::to_string(wavefront_size);

//	cl::cl_context_properties
//	OCL_SAFE_CALL( clGetEventProfilingInfo(evfirst, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
//	OCL_SAFE_CALL( clGetEventProfilingInfo(evlast, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	cl::Context context = cl::Context(VECTOR_CLASS<cl::Device>(1, dev));  
	cl::CommandQueue queue = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE);  

	std::ifstream t("reduction_kernels.cl");
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string src_string = buffer.str();

	std::cout << "Building with options \"" << cl_version_option << "\"" << std::endl;
/*	std::cout << "------- SOURCE --------" << std::endl;
	std::cout << src_string << std::endl;
	std::cout << "-----------------------" << std::endl;*/

	cl::Program::Sources src(1, std::make_pair(src_string.c_str(), src_string.size()));  
	cl::Program program(context, src);
//	std::cout << "Building OpenCL program" << std::endl;
	try{
		program.build(VECTOR_CLASS<cl::Device>(1, dev), cl_version_option.c_str());
	} catch(cl::Error & e){
		std::string buildLog; 
		program.getBuildInfo(dev, CL_PROGRAM_BUILD_LOG, &buildLog);  
		std::cout << "Build log:" << std::endl  
			<< " ******************** " << std::endl  
			<< buildLog << std::endl  
			<< " ******************** " << std::endl;
		throw;
	}

	std::string buildLog; 
	program.getBuildInfo(dev, CL_PROGRAM_BUILD_LOG, &buildLog);
	if( buildLog!="" )
		std::cout << "Build log:" << std::endl  
			<< " ******************** " << std::endl  
			<< buildLog << std::endl  
			<< " ******************** " << std::endl;

	cl::Kernel kernel1(program, "reductionShmem");
	cl::NDRange globalSize(64*4*256), localSize(256);
	cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));

//	size_t workgroup_size;
//	kernel1.getWorkGroupInfo(dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &workgroup_size);
//	std::cout << "wkgroup mult:   " << workgroup_size << std::endl;

//	std::cout << "Initializing result buffer" << std::endl;
	cl_uint* map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));
//	std::cout << (intptr_t)map << std::endl;
	map[0] = 0;
	queue.enqueueUnmapMemObject(buff, map);
//	std::cout << "NDRange size " << globalSize[0] << std::endl;

	unsigned int validResult = globalSize[0]*(globalSize[0]-1)/2;
//	std::cout << "valid result " << validResult << std::endl;

	kernel1.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
	kernel1.setArg(1, buff);

	cl::Event eKernel;
//eKernel.
	// Shared memory kernel
	std::cout << std::endl << "1. Shared memory kernel" << std::endl;

	std::cout << "Executing...";
	queue.enqueueNDRangeKernel(kernel1, cl::NullRange, globalSize, localSize, NULL, &eKernel);
	queue.finish();
	std::cout << "Done!" << std::endl;
	auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	double elapsedTime1 = (infFinish-infStart)/1000000.0;

	map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));
	std::cout << "Output: " << map[0] << " in " << elapsedTime1 << " msecs" << std::endl;
	if( map[0] == validResult )
		std::cout << "PASSED!" << std::endl;
	else
		std::cout << "FAILED (" << map[0] << "!=" << validResult << ")!" << std::endl;
	queue.enqueueUnmapMemObject(buff, map);  

	std::cout << std::endl << "2. Hybrid kernel via subgroup functions" << std::endl;
	if( cl_subgroups ){
//		std::cout << "Done!" << std::endl;
		cl::Kernel kernel2(program, "reductionSubgrp");
		map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));
		map[0] = 0;
		queue.enqueueUnmapMemObject(buff, map);
		kernel2.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
		kernel2.setArg(1, buff);

//		cl::Event eKernel;
		std::cout << "Executing...";
		queue.enqueueNDRangeKernel(kernel2, cl::NullRange, globalSize, localSize, NULL, &eKernel);
		queue.finish();
		std::cout << "Done!" << std::endl;
		auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		double elapsedTime2 = (infFinish-infStart)/1000000.0;

		map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));
		std::cout << "Output: " << map[0] << " in " << elapsedTime2 << " msecs (relative speed-up to kernel 1: " << elapsedTime1/elapsedTime2 << ")" << std::endl;
		if( map[0] == validResult )
			std::cout << "PASSED!" << std::endl;
		else
			std::cout << "FAILED (" << map[0] << "!=" << validResult << ")!" << std::endl;
		queue.enqueueUnmapMemObject(buff, map);  
	} else
		std::cout << "Subgroups not supported. Skipping kernel 2." << std::endl;
	std::cout << std::endl << "3. Workgroup function kernel" << std::endl;
	if( cl_ver20 ){
//		std::cout << "Done!" << std::endl;
		cl::Kernel kernel3(program, "reductionWkgrp");
		map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));
		map[0] = 0;
		queue.enqueueUnmapMemObject(buff, map);
		kernel3.setArg(0, buff);

//		cl::Event eKernel;
		std::cout << "Executing...";
		queue.enqueueNDRangeKernel(kernel3, cl::NullRange, globalSize, localSize, NULL, &eKernel);
		queue.finish();
		std::cout << "Done!" << std::endl;
		auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		double elapsedTime3 = (infFinish-infStart)/1000000.0;

		map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));
		std::cout << "Output: " << map[0] << " in " << elapsedTime3 << " msecs (relative speed-up to kernel 1: " << elapsedTime1/elapsedTime3 << ")" << std::endl;
		if( map[0] == validResult )
			std::cout << "PASSED!" << std::endl;
		else
			std::cout << "FAILED (" << map[0] << "!=" << validResult << ")!" << std::endl;
		queue.enqueueUnmapMemObject(buff, map);  
	} else
		std::cout << "OpenCL 2.0 is not supported. Skipping kernel 3." << std::endl;


//	map = (cl_uint*)queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));  
//	std::cout << "res = " << map[0] << std::endl
//		<< "kernel execution time = " << elapsedTime << " msecs" << std::endl;
//	queue.enqueueUnmapMemObject(buff, map);  

	return 0;  
}
