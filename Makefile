NVCC = nvcc
GCC = g++

CU_SRC = offline.cu
CPP_SRC = main.cpp
CU_OBJ = offline.o
CPP_OBJ = main.o
EXEC = main

all: $(EXEC)

$(EXEC): $(CPP_OBJ) $(CU_OBJ)
	$(NVCC) $(CPP_OBJ) $(CU_OBJ) -o $(EXEC) -lcuda -lnvrtc -lnvJitLink

$(CU_OBJ): $(CU_SRC)
	$(NVCC) -rdc=true -c $< -o $@

$(CPP_OBJ): $(CPP_SRC)
	$(GCC) -c $< -o $@

clean:
	rm -f $(CPP_OBJ) $(CU_OBJ) $(PTX_FILE) $(EXEC)