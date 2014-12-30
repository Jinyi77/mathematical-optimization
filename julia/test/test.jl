include("../src/nelder_mead.jl")

function rosenbrock(x::Vector, a::Float64 = 1.0, b::Float64 = 100.0)
    return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

init_point = [3.0, 5.0]

eval, p = nelder_mead(rosenbrock, init_point)
