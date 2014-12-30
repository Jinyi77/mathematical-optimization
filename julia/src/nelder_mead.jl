#-------------------------------------------------------------------------------
# References:
# http://www.scholarpedia.org/article/Nelder-Mead_algorithm
# http://nbviewer.ipython.org/github/QuantEcon/QuantEcon.site/blob/master/
#        _static/notebooks/chase_nelder_mead.ipynb
#-------------------------------------------------------------------------------

function nelder_mead(obj_fun::Function,
                     init_point::Vector,
                     tol::Float64 = 1e-8,
                     max_itr::Int64 = 1000,
                     init_increments::Vector = ones(length(init_point)),
                     α::Float64 = 1.0,
                     γ::Float64 = 2.0,
                     ρ::Float64 = -0.5,
                     σ::Float64 = 0.5)
    itr = 0
    n = length(init_point)
    p = repmat(init_point, 1, n+1)
    for i = 1:n
        p[i, i] += init_increments[i]
    end

    while true
        evals = Float64[obj_fun(vec(p[:, j])) for j =1:3]

        itr += 1
        # Step1: Ordering
        p = p[:, sortperm(evals)]

        # Step2: Calculating centroid
        p_o = vec(mean(p[:, 1:end-1], 2))

        # Step3: Reflection
        p_r = p_o + α * (p_o - vec(p[:, end]))
        if obj_fun(vec(p[:, 1])) <= obj_fun(p_r) <obj_fun(vec(p[:, end-1]))
            p[:, end] = deepcopy(p_r)
        # Step3.1: Expansion
        elseif obj_fun(p_r) < obj_fun(vec(p[:, 1]))
            p_e = p_o + γ *(p_o - vec(p[:, end]))
            if obj_fun(p_e) < obj_fun(p_r)
                p[:, end] = deepcopy(p_e)
            else
                p[:, end] = deepcopy(p_r)
            end
        # Step3.2: Contraction
        else
            p_c = p_o + ρ * (p_o - vec(p[:, end]))
            if obj_fun(p_c) < obj_fun(vec(p[:, end]))
                p[:, end] = deepcopy(p_c)
            # Step3.2.1: Reduction
            else
              p[:, 2:end] = deepcopy(broadcast(+, (σ - 1) * vec(p[:, 1]),
                                               p[:, 2:end]))
            end
        end

        if sqrt(var(evals) * n / (n+1)) < tol
            println("Algorithm converges with $itr iterations")
            return (obj_fun(vec(mean(p, 2))), mean(p, 2))
        end

        if itr == max_itr
            println("Reach the max number of iterations($itr ) before converge")
            return (obj_fun(vec(mean(p, 2))), mean(p, 2))
        end
    end

end
