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

double executeKernel(const cl::CommandQueue &queue, const cl::Kernel &kernel, const cl::NDRange &globalRange, const cl::NDRange &localRange){
	cl::Event eKernel;
	std::cout << "Executing...";
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange, NULL, &eKernel);
	queue.finish();
	std::cout << "Done!" << std::endl;
	auto infStart = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	auto infFinish = eKernel.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (infFinish-infStart)/1000000.0;
}

cl_uint readBufferData(cl::CommandQueue queue, cl::Buffer buffer){
	cl_uint *map = (cl_uint*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint));
	cl_uint result = map[0];
	queue.enqueueUnmapMemObject(buffer, map);
	return result;
}

void writeBufferData(cl::CommandQueue queue, cl::Buffer buffer, cl_uint value){
	cl_uint *map = (cl_uint*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint));
	map[0] = value;
	queue.enqueueUnmapMemObject(buffer, map);
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

int main(void) {  
	std::cout << "Workgroup and sub-workgroup OpenCL 2.0 function evaluation test case" << std::endl;
	cl::Platform pl;
	cl::Device dev;
	selectPlatformDevice(pl, dev);

	cl_device_type devType;
	dev.getInfo(CL_DEVICE_TYPE, &devType);

	std::cout << std::endl << "Device info"<< std::endl;
	int wavefront_size=1;
	std::string sInfo;
	pl.getInfo(CL_PLATFORM_NAME, &sInfo);
	std::cout << "Platform:       " << trim(sInfo) << std::endl;
	if( devType==CL_DEVICE_TYPE_GPU ){
		if( sInfo.find("AMD") != std::string::npos )
			wavefront_size = 64;
		if( sInfo.find("NVIDIA") != std::string::npos )
			wavefront_size = 32;
	}

	dev.getInfo(CL_DEVICE_NAME, &sInfo);  
	std::cout << "Device:         " << trim(sInfo) << std::endl;
	dev.getInfo(CL_DRIVER_VERSION, &sInfo);
	std::cout << "Driver version: " << trim(sInfo) << std::endl;

	dev.getInfo(CL_DEVICE_VERSION, &sInfo);
	std::cout << "OpenCL version: " << trim(sInfo) << std::endl;
    size_t vStart = sInfo.find(' ', 0), vEnd = sInfo.find('.', 0);
	int CLMajorVer = std::stoi( sInfo.substr(vStart + 1, vEnd - vStart - 1) ), CLMinorVer = std::stoi( sInfo.substr(vEnd + 1, 1) );
	std::string cl_version_option;
	bool cl_subgroups = false, cl_ver20 = false;
	if( CLMajorVer>=2 ) {
		std::cout << "Great! OpenCL 2.0 is supported :)" << std::endl;
		cl_ver20 = true;
		cl_version_option = "-cl-std=CL2.0 -cl-uniform-work-group-size -DK3";
		dev.getInfo(CL_DEVICE_EXTENSIONS, &sInfo);
		std::string extension;
		std::stringstream ssInfo(sInfo);
		while( ssInfo >> extension )
			if( extension=="cl_khr_subgroups" )
				cl_subgroups = true;
		if( cl_subgroups )
			cl_version_option += " -DK2";
	} else {
		std::cout << "Using OpenCL 1." << CLMinorVer << std::endl;
		cl_version_option = "-cl-std=CL1."+std::to_string(CLMinorVer);
	}
	if( wavefront_size>1 )
		cl_version_option += " -DWAVEFRONT_SIZE="+std::to_string(wavefront_size);

	// Create context, queue
	cl::Context context = cl::Context(VECTOR_CLASS<cl::Device>(1, dev));  
	cl::CommandQueue queue = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE);  

	// Load and build kernel
	std::ifstream t("reduction_kernels.cl");
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string src_string = buffer.str();
	std::cout << "Building kernel with options \"" << cl_version_option << "\"" << std::endl;
	cl::Program::Sources src(1, std::make_pair(src_string.c_str(), src_string.size()));  
	cl::Program program(context, src);
	try {
		program.build(VECTOR_CLASS<cl::Device>(1, dev), cl_version_option.c_str());
	} catch(cl::Error&){
		std::string buildLog; 
		program.getBuildInfo(dev, CL_PROGRAM_BUILD_LOG, &buildLog);  
		std::cout << "Build log:" << std::endl  
			<< " ******************** " << std::endl  
			<< buildLog << std::endl  
			<< " ******************** " << std::endl;
		throw;
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
	cl::NDRange globalSize(64*4*256), localSize(256);
	cl::Buffer buff(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl::Event eKernel;

	//	Initialize result buffer
	writeBufferData(queue, buff, 0);
//	std::cout << "NDRange size " << globalSize[0] << std::endl;

	// Precalc correct result
	unsigned int validResult = globalSize[0]*(globalSize[0]-1)/2;
//	std::cout << "valid result " << validResult << std::endl;

	kernel1.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
	kernel1.setArg(1, buff);

	// Shared memory kernel
	std::cout << std::endl << "1. Shared memory kernel" << std::endl;
	double elapsedTime1 = executeKernel(queue, kernel1, globalSize, localSize);

	// Verify result
	cl_uint result = readBufferData(queue, buff);
	outputInfo(result, elapsedTime1, globalSize[0]);
	verifyResult(result, validResult);

	// Subgroup function kernel
	std::cout << std::endl << "2. Hybrid kernel via subgroup functions" << std::endl;
	if( cl_subgroups ){
		cl::Kernel kernel2(program, "reductionSubgrp");
		writeBufferData(queue, buff, 0);

		kernel2.setArg(0, localSize[0]*sizeof(cl_uint), NULL);
		kernel2.setArg(1, buff);
		double elapsedTime2 = executeKernel(queue, kernel2, globalSize, localSize);

		cl_uint result = readBufferData(queue, buff);
		outputInfo(result, elapsedTime2, globalSize[0]);
		std::cout << "Relative speed-up to kernel 1: " << elapsedTime1/elapsedTime2 << std::endl;
		verifyResult(result, validResult);
	} else
		std::cout << "Subgroups not supported. Skipping kernel 2." << std::endl;

	// Workgroup function kernel
	std::cout << std::endl << "3. Workgroup function kernel" << std::endl;
	if( cl_ver20 ){
		cl::Kernel kernel3(program, "reductionWkgrp");
		writeBufferData(queue, buff, 0);

		kernel3.setArg(0, buff);
		double elapsedTime3 = executeKernel(queue, kernel3, globalSize, localSize);

		cl_uint result = readBufferData(queue, buff);
		outputInfo(result, elapsedTime3, globalSize[0]);
		std::cout << "Relative speed-up to kernel 1: " << elapsedTime1/elapsedTime3 << std::endl;
		verifyResult(result, validResult);
	} else
		std::cout << "OpenCL 2.0 is not supported. Skipping kernel 3." << std::endl;

	return 0;  
}
