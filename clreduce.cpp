#define __CL_ENABLE_EXCEPTIONS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>  
#include <CL/cl.hpp>

#define WG_SIZE 256
#define _MAKE_STR(A) #A
#define MAKE_STR(A) _MAKE_STR(A)

inline std::string trim(std::string str) {
	str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
	str.erase(str.find_last_not_of(' ')+1);         //surfixing spaces
	return str; // use substring...
}

// Platform and device selection
void selectPlatformDevice(cl::Platform &platform, cl::Device &device){
	VECTOR_CLASS<cl::Platform> platforms;
	VECTOR_CLASS<cl::Device> devices;
	std::cout << "Platform/Device selection" << std::endl;
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
	// Platform selection
	if( platforms.size()>1 ){
		std::cout << "Select platform index: ";
		std::cin >> iP;
		while( iP<0 || iP>(int)platforms.size() ){
			std::cout << "Invalid platform index. Select again:";
			std::cin >> iP;
		}
	} else
		iP = 1;
	platform = platforms[iP-1];
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	// Device selection
	if( devices.size()>1 ){
		std::cout << "Select device index: ";
		std::cin >> iD;
		while( iD<0 || iD>(int)devices.size() ){
			std::cout << "Invalid device index. Select again:";
			std::cin >> iD;
		}
	} else
		iD = 1;
	device = devices[iD-1];
}

void writeBufferData(cl::CommandQueue queue, cl::Buffer buffer, cl_uint value){
	queue.enqueueFillBuffer<cl_uint>(buffer, value, 0, sizeof(value), NULL, NULL);
	queue.finish();
}

double executeKernel(const cl::CommandQueue &queue, cl::Buffer buffer, const cl::Kernel &kernel, const cl::NDRange &globalRange, const cl::NDRange &localRange){
	const int TOTAL_ITS = 100;
	double total_time = 0.;
	std::cout << "Executing...";
	for(int i=0; i<TOTAL_ITS; i++){
		cl::Event eKernel;
		// Initialize result buffer
		writeBufferData(queue, buffer, 0);
		// Kernel execution
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange, NULL, &eKernel);
		queue.finish();
		auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		total_time += (infFinish-infStart)/1000000.;
	}
	std::cout << "Done!" << std::endl;
	return total_time/TOTAL_ITS;
}

cl_uint readBufferData(cl::CommandQueue queue, cl::Buffer buffer){
	cl_uint *map = (cl_uint*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));
	cl_uint result = map[0];
	queue.enqueueUnmapMemObject(buffer, map);
	return result;
}

void verifyResult(cl_uint result, unsigned int validResult){
	if( result == validResult )
		std::cout << "PASSED!" << std::endl;
	else
		std::cout << "FAILED (" << result << "!=" << validResult << ")!" << std::endl;
}

void outputInfo(unsigned int result, double elapsedTime, int workitems){
	std::cout << "Output: " << result << " / Time: " << elapsedTime << " msecs (" << (workitems/elapsedTime/1000000.0) << " billion elements/second)" << std::endl;
}

void try_build_program(cl::Program program, cl::Device dev, const std::string &str_cl_parameters){
	std::cout << "Building kernel with options \"" << str_cl_parameters << "\"" << std::endl;
	try {
		program.build(VECTOR_CLASS<cl::Device>(1, dev), str_cl_parameters.c_str());
	} catch(cl::Error&){
		std::string buildLog; 
		program.getBuildInfo(dev, CL_PROGRAM_BUILD_LOG, &buildLog);  
		std::cout << "Build log:" << std::endl  
			<< " ******************** " << std::endl  
			<< buildLog << std::endl  
			<< " ******************** " << std::endl;
		throw;
	}
}

int main(void) {  
	std::cout << "Workgroup and sub-workgroup OpenCL 2.0 function evaluation test case" << std::endl;
	cl::Platform pl;
	cl::Device dev;
	selectPlatformDevice(pl, dev);

	cl_device_type devType;
	dev.getInfo(CL_DEVICE_TYPE, &devType);

	std::cout << std::endl << "Device info"<< std::endl;
	std::string sInfo;
	pl.getInfo(CL_PLATFORM_NAME, &sInfo);
	std::cout << "Platform:       " << trim(sInfo) << std::endl;

	dev.getInfo(CL_DEVICE_NAME, &sInfo);  
	std::cout << "Device:         " << trim(sInfo) << std::endl;
	dev.getInfo(CL_DRIVER_VERSION, &sInfo);
	std::cout << "Driver version: " << trim(sInfo) << std::endl;

	dev.getInfo(CL_DEVICE_OPENCL_C_VERSION, &sInfo);
	std::cout << "OpenCL C version: " << trim(sInfo) << std::endl;
	size_t vEnd = sInfo.find('.', 0);
	size_t vStart = sInfo.rfind(' ', vEnd);
	int CLMajorVer = std::stoi( sInfo.substr(vStart + 1, vEnd - vStart - 1) ), CLMinorVer = std::stoi( sInfo.substr(vEnd + 1, 1) );
	std::string str_cl_parameters("-DWG_SIZE=" MAKE_STR(WG_SIZE) " -Werror ");
	bool cl_subgroups = false, cl_ver20 = false;
	if( CLMajorVer>=2 ) {
		std::cout << "Great! OpenCL 2.0 is supported :)" << std::endl;
		cl_ver20 = true;
		str_cl_parameters += "-cl-std=CL2.0 -cl-uniform-work-group-size -DK3";
		dev.getInfo(CL_DEVICE_EXTENSIONS, &sInfo);
		std::string extension;
		std::stringstream ssInfo(sInfo);
		while( ssInfo >> extension )
			if( extension=="cl_khr_subgroups" || extension=="cl_intel_subgroups" )
				cl_subgroups = true;
		if( cl_subgroups )
			str_cl_parameters += " -DK2";
	} else {
		std::cout << "Using OpenCL 1." << CLMinorVer << std::endl;
		str_cl_parameters += "-cl-std=CL1."+std::to_string(CLMinorVer);
	}

	// Create context, queue
	cl::Context context = cl::Context(VECTOR_CLASS<cl::Device>(1, dev));  
	cl::CommandQueue queue = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE);  

	// Load and build kernel
	std::ifstream t("reduction_kernels.cl");
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string src_string = buffer.str();
	cl::Program::Sources src(1, std::make_pair(src_string.c_str(), src_string.size()));  
	cl::Program program(context, src);

	try_build_program(program, dev, str_cl_parameters);
	size_t wavefront_size_old = 0, wavefront_size = 1;
	while(wavefront_size_old != wavefront_size) {
		wavefront_size_old = wavefront_size;
		cl::Kernel k(program, "reductionShmem");
		k.getWorkGroupInfo(dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &wavefront_size);

		if( wavefront_size_old!=wavefront_size ) {
			std::cout << "Warp/Wavefront size determined: " << wavefront_size << std::endl;
			std::string str_cl_parameters_wavefront(str_cl_parameters);
			str_cl_parameters_wavefront += " -DWAVEFRONT_SIZE="+std::to_string(wavefront_size);
			program = cl::Program{context, src}; // recreate program as some platforms throw exception after calling build twice
			try_build_program(program, dev, str_cl_parameters_wavefront);
		}
	}

	// Print built log if not empty
	std::string buildLog; 
	program.getBuildInfo(dev, CL_PROGRAM_BUILD_LOG, &buildLog);
	if( buildLog!="" )
		std::cout << "Build log:" << std::endl  
			<< " ******************** " << std::endl  
			<< buildLog << std::endl  
			<< " ******************** " << std::endl;

	// Create kernel and set NDRange size
	cl::Kernel kernel1(program, "reductionShmem");
	int MAX_CUs = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	cl::NDRange globalSize(MAX_CUs*1024*WG_SIZE), localSize(WG_SIZE);
	std::cout << "Launching NDRange size of " << static_cast<const ::size_t*>(globalSize)[0]/WG_SIZE << " workgroups with " << WG_SIZE << " workitems per workgroup" << std::endl;
	cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl::Event eKernel;

	// Precalc correct result
	unsigned int validResult = globalSize[0]*(globalSize[0]-1)/2;
//	std::cout << "valid result " << validResult << std::endl;

	kernel1.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
	kernel1.setArg(1, buff);

	// Shared memory kernel
	std::cout << std::endl << "1. Shared memory only kernel" << std::endl;
	double elapsedTime1 = executeKernel(queue, buff, kernel1, globalSize, localSize);

	// Verify result
	cl_uint result = readBufferData(queue, buff);
	outputInfo(result, elapsedTime1, globalSize[0]);
	verifyResult(result, validResult);

	// Subgroup function kernel
	std::cout << std::endl << "2. Hybrid kernel via subgroup functions" << std::endl;
	if( cl_subgroups ){
		cl::Kernel kernel2(program, "reductionSubgrp");

		kernel2.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
		kernel2.setArg(1, buff);
		double elapsedTime2 = executeKernel(queue, buff, kernel2, globalSize, localSize);

		cl_uint result = readBufferData(queue, buff);
		outputInfo(result, elapsedTime2, globalSize[0]);
		std::cout << "Relative speed-up with respect to kernel 1: " << elapsedTime1/elapsedTime2 << " (" << elapsedTime2/elapsedTime1 << " times slower)" << std::endl;
		verifyResult(result, validResult);
	} else
		std::cout << "Subgroups not supported. Skipping kernel 2." << std::endl;

	// Workgroup function kernel
	std::cout << std::endl << "3. Workgroup function kernel" << std::endl;
	if( cl_ver20 ){
		cl::Kernel kernel3(program, "reductionWkgrp");

		kernel3.setArg(0, buff);
		double elapsedTime3 = executeKernel(queue, buff, kernel3, globalSize, localSize);

		cl_uint result = readBufferData(queue, buff);
		outputInfo(result, elapsedTime3, globalSize[0]);
		std::cout << "Relative speed-up with respect to kernel 1: " << elapsedTime1/elapsedTime3 << " (" << elapsedTime3/elapsedTime1 << " times slower)" << std::endl;
		verifyResult(result, validResult);
	} else
		std::cout << "OpenCL 2.0 is not supported. Skipping kernel 3." << std::endl;

	return 0;  
}
