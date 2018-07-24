################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../GPUroutines.cu \
../Las.cu \
../complexExtension.cu \
../main.cu 

CU_DEPS += \
./GPUroutines.d \
./Las.d \
./complexExtension.d \
./main.d 

OBJS += \
./GPUroutines.o \
./Las.o \
./complexExtension.o \
./main.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


