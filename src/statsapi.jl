# ============================================================
# SpectralGLM
# ============================================================

StatsAPI.coef(m::SpectralGLM) = m.coef
StatsAPI.response(m::SpectralGLM) = m.y
StatsAPI.modelmatrix(m::SpectralGLM) = m.X
StatsAPI.nobs(m::SpectralGLM) = length(m.y)

StatsModels.coefnames(m::SpectralGLM) =
    m.formula === nothing ?
    ["x$(j)" for j in eachindex(m.coef)] :
    StatsModels.coefnames(m.formula.rhs)

function StatsAPI.predict(m::SpectralGLM, Xnew::AbstractMatrix)
    η = Xnew * StatsAPI.coef(m)
    return GLM.linkinv.(Ref(m.link), η)
end

function StatsAPI.predict(m::SpectralGLM, newdata)
    m.formula === nothing &&
        throw(ArgumentError(
            "table prediction requires a model fitted with the formula interface"
        ))

    Xnew = StatsModels.modelcols(m.formula.rhs, newdata)
    return StatsAPI.predict(m, Xnew)
end

StatsAPI.predict(m::SpectralGLM) =
    StatsAPI.predict(m, m.X)

StatsAPI.fitted(m::SpectralGLM) =
    StatsAPI.predict(m)


# ============================================================
# SpectralModel
# ============================================================

StatsAPI.coef(m::SpectralModel) = m.coef
StatsAPI.response(m::SpectralModel) = m.y
StatsAPI.modelmatrix(m::SpectralModel) = m.X
StatsAPI.nobs(m::SpectralModel) = length(m.y)

StatsModels.coefnames(m::SpectralModel) =
    ["x$(j)" for j in eachindex(m.coef)]

StatsAPI.predict(m::SpectralModel, Xnew::AbstractMatrix) =
    Xnew * StatsAPI.coef(m)

StatsAPI.predict(m::SpectralModel) =
    StatsAPI.predict(m, m.X)

StatsAPI.fitted(m::SpectralModel) =
    StatsAPI.predict(m)