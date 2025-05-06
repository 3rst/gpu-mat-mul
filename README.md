# gpu-mat-mul
Trying to beat cuBLAS in matrix multiply on a GPU

Final timing information

Block size: 16  
input: 5000x5000  
Total time for naive kernel: 1.003580, copy to gpu: 0.043073, kernel: 0.867438, copy from gpu: 0.093069  
Total time for optimized kernel: 0.294244, copy to gpu: 0.043073, kernel: 0.228204, copy from gpu: 0.022967  
Total time for cuBLAS kernel: 0.098762, copy to and kernel: 0.075724, copy from GPU: 0.023038  
  
Speedup: 3.4 times  
  
input: 10000x10000  
Total time for naive kernel: 7.919882, copy to gpu: 0.171052, kernel: 7.374920, copy from gpu: 0.373910  
Total time for optimized kernel: 2.074616, copy to gpu: 0.171052, kernel: 1.813229, copy from gpu: 0.090335  
Total time for cuBLAS kernel: 0.503005, copy to and kernel: 0.412316, copy from GPU: 0.090689  

Speedup: 3.8 times  

input: 15000x15000  
Total time for naive kernel: 29.990614, copy to gpu: 0.381556, kernel: 28.755886, copy from gpu: 0.853173  
Total time for optimized kernel: 6.712379, copy to gpu: 0.381556, kernel: 6.126357, copy from gpu: 0.204466  
Total time for cuBLAS kernel: 1.421119, copy to and kernel: 1.216489, copy from GPU: 0.204630  
  
Speedup: 4.4 times  
  
input: 20000x20000  
Total time for naive kernel: 67.166992, copy to gpu: 0.683246, kernel: 64.992935, copy from gpu: 1.490818  
Total time for optimized kernel: 15.548171, copy to gpu: 0.683246, kernel: 14.496979, copy from gpu: 0.367947  
Total time for cuBLAS kernel: 3.044753, copy to and kernel: 2.679869, copy from GPU: 0.364884  
  
Speedup: 4.3  



Block size: 32  
input: 5000x5000  
Total time for naive kernel: 1.172673, copy to gpu: 0.043207, kernel: 1.033137, copy from gpu: 0.096329  
Total time for optimized kernel: 0.250535, copy to gpu: 0.043207, kernel: 0.184249, copy from gpu: 0.023079  
Total time for cuBLAS kernel: 0.098716, copy to and kernel: 0.075683, copy from GPU: 0.023033  
  
Speedup: 4.6 times  
  
input: 10000x10000  
Total time for naive kernel: 9.227868, copy to gpu: 0.170268, kernel: 8.685626, copy from gpu: 0.371974  
Total time for optimized kernel: 1.716788, copy to gpu: 0.170268, kernel: 1.455953, copy from gpu: 0.090567  
Total time for cuBLAS kernel: 0.503229, copy to and kernel: 0.412898, copy from GPU: 0.090331  
  
Speedup: 5.3 times  
  
input: 15000x15000  
Total time for naive kernel: 34.277794, copy to gpu: 0.395932, kernel: 33.044060, copy from gpu: 0.837801  
Total time for optimized kernel: 5.498927, copy to gpu: 0.395932, kernel: 4.898998, copy from gpu: 0.203997  
Total time for cuBLAS kernel: 1.423728, copy to and kernel: 1.220466, copy from GPU: 0.203262  
  
Speedup: 6.2 times  
  
input: 20000x20000  
Total time for naive kernel: 66.629066, copy to gpu: 0.692185, kernel: 64.451134, copy from gpu: 1.485752  
Total time for optimized kernel: 12.620554, copy to gpu: 0.692185, kernel: 11.565141, copy from gpu: 0.363228  
Total time for cuBLAS kernel: 3.038954, copy to and kernel: 2.664773, copy from GPU: 0.374181  
  
Speedup: 5.2 times  