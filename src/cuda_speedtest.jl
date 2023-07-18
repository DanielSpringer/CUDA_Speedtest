module cuda_speedtest

greet() = print("Hello World!")

using Pkg
Pkg.add("CUDA")
Pkg.activate(".")
Pkg.instantiate()
using CUDA
CUDA.allowscalar(true)
print("**************************************** \n")
print(CUDA.functional())
print("\n")
print(CUDA.versioninfo())
print("\n")
Pkg.add("Einsum")
using Einsum

function cuda1(N)
    # M = [1 2; 2 1]
    # M2 = [1 2; 2 1] #[1; 2;; 2; 1;;;1; 3;; 3; 1]
    # Z = CUDA.zeros(2,2)
    M = rand(N,N)
    M2 = rand(N,N)
    M_c = CuArray(M)
    M2_c = CuArray(M2)
    Z = CUDA.similar(M)
    @einsum Z[v1,v2] = M_c[v1,v2] + M2_c[v1,v2]
    # print(Z)
end

function cuda2(N)
    # M = [1 2; 2 1]
    # M2 = [1 2; 2 1] #[1; 2;; 2; 1;;;1; 3;; 3; 1]
    # Z = CUDA.zeros(2,2)
    M = rand(N,N)
    M2 = rand(N,N)
    M_c = CuArray(M)
    M2_c = CuArray(M2)
    Z = CUDA.similar(M)
    Z = M_c * M2_c
    # print(Z)
end


function cpu1(N)
    # M = [1 2; 2 1]
    # M2 = [1 2; 2 1] #[1; 2;; 2; 1;;;1; 3;; 3; 1]
    # Z = zeros(2,2)
    M = rand(N,N)
    M2 = rand(N,N)
    Z = similar(M)
    @einsum Z[v1,v2] = M[v1,v2] + M2[v1,v2]
    # print(Z)
end

print("***************************************** \n")
print("***************************************** \n")
print("***************************************** \n")
Pkg.add("BenchmarkTools")
using BenchmarkTools
N = 1000
@btime cuda1(N)
@btime cuda2(N)
@btime cpu1(N)


# id = [1 0; 0 1]
# g = zeros(2,2)
# @einsum g[v1,v2] = id[v1,i1] * M[i1,i2] + id[v1,v2]
# @einsum g[v1,v2] = (id[v1,i1] * M[i1,v2]) + id[v1,v2]


end # module
