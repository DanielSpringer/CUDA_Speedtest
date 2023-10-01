import Pkg; 
Pkg.activate("../")
Pkg.precompile()
Pkg.add("CUDA")
Pkg.instantiate()
using LinearAlgebra
using CUDA
CUDA.allowscalar(true)
print("**************************************** \n")
print(CUDA.functional())
print("\n")
print(CUDA.versioninfo())
print("\n")

function cuda_inv(N)
    M = rand(N,N)
    id = Matrix(1I, size(M)[1], size(M)[2])
    M_c = CUDA.CuArray(M)
    id_c = CUDA.CuArray(id)
    i_M = id_c / M_c
    # i_M = inv(M_c)
    # print(diag(i_M * M_c))
end

function cpu_inv(N)
    M = rand(N,N)
    id = Matrix(1I, size(M)[1], size(M)[2])
    i_M = id / M
    # i_M = inv(M)
    # print(diag(i_M * M_c))
end

Pkg.add("BenchmarkTools")
using BenchmarkTools
N = 50

# print("CPU N 50")
# N = 50
# @btime cpu_inv(N)

# print("CPU N 100")
# N = 100
# @btime cpu_inv(N)

# print("CPU N 200")
# N = 200
# @btime cpu_inv(N)

# print("CPU N 500")
# N = 500
# @btime cpu_inv(N)

print("CPU N 700")
N = 700
@btime cpu_inv(N)




# print("CUDA N 50")
# N = 50
# @btime cuda_inv(N)

# print("CUDA N 100")
# N = 100
# @btime cuda_inv(N)

# print("CUDA N 200")
# N = 200
# @btime cuda_inv(N)

# print("CUDA N 500")
# N = 500
# @btime cuda_inv(N)

print("CUDA N 700")
N = 700
@btime cuda_inv(N)

