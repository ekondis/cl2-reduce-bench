#OCLSDKDIR = /usr/local/cuda
OCLSDKDIR = ${AMDAPPSDKROOT}
CPP = g++
OCLSDKINC = ${OCLSDKDIR}/include
OCLSDKLIB = ${OCLSDKDIR}/lib/x86_64
OPTFLAG =  -fomit-frame-pointer
#INCLUDES = -I../common/inc
FLAGS = -std=c++11 -O2 ${INCLUDES} -I${OCLSDKINC} 
LFLAGS = -L${OCLSDKLIB} 
LIBPARS = -lOpenCL -lrt

.PHONY: clean all

all: cl2-reduce-bench

cl2-reduce-bench: clreduce.o
	${CPP} ${LFLAGS} -o $@ $< ${LIBPARS}

clreduce.o: clreduce.cpp
	${CPP} -c ${FLAGS} $<

clean:
	rm cl2-reduce-bench clreduce.o
