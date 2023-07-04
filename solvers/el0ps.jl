using El0ps

function run_el0ps(X, y, M, λ, tolerance)

    f = LeastSquares(y)
    h = Bigm(M)
    λ = λ / length(y)
    problem = Problem(f, h, X, λ)
    solver = BnbSolver(
        maxtime=Inf,
        tolgap=Float64(tolerance),
        tolint=1e-6,
        dualpruning=true,
        l0screening=true,
        l1screening=true,
    )
    result = optimize(solver, problem)
    return result.x
end
