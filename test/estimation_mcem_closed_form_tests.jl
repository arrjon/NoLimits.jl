using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

function _simple_normal_re_model()
    @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
end

function _simple_normal_re_dm()
    model = _simple_normal_re_model()
    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t  = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y  = [0.1, 0.2, 0.0, -0.1, 0.3, 0.4]
    )
    DataModel(model, df; primary_id=:ID, time_col=:t)
end

# ---------------------------------------------------------------------------
# Test 1: closed-form is used for Normal RE with MCMC E-step
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: Normal RE with MCMC E-step" begin
    dm = _simple_normal_re_dm()
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
        maxiters=3
    ))
    @test res isa FitResult
    @test NoLimits.get_closed_form_mstep_used(res) == true
    notes = NoLimits.get_notes(res)
    @test notes isa NamedTuple
    @test hasproperty(notes, :closed_form_mstep_used)
    @test hasproperty(notes, :closed_form_mstep_mode)
    @test hasproperty(notes, :closed_form_builtin_eligibility)
    # The RE std dev is handled by closed-form; 'a' is numeric → :hybrid
    @test notes.closed_form_mstep_mode == :hybrid
end

# ---------------------------------------------------------------------------
# Test 2: closed-form with IS E-step (weighted moments)
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: Normal RE with IS E-step" begin
    dm = _simple_normal_re_dm()
    res = fit_model(dm, NoLimits.MCEM(;
        e_step=NoLimits.MCEM_IS(n_samples=250),
        maxiters=3
    ))
    @test res isa FitResult
    @test NoLimits.get_closed_form_mstep_used(res) == true
    notes = NoLimits.get_notes(res)
    @test notes.closed_form_mstep_mode == :hybrid
end

# ---------------------------------------------------------------------------
# Test 3: all parameters closed-form → optimizer is skipped
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: all params closed-form" begin
    # Model where every fixed-effect parameter maps to a supported closed-form target:
    # σ_η (RE std dev of Normal(0, σ_η)) — closed-form via re_cov_params.
    # No other free params → closed_form_only.
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            σ = RealNumber(0.8, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ); column=:ID)
        end
        @formulas begin
            y ~ Normal(η, 0.3)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t  = [0.0, 1.0, 0.0, 1.0],
        y  = [0.1, 0.2, -0.1, 0.0]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
        maxiters=3
    ))
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.closed_form_mstep_used == true
    @test notes.closed_form_mstep_mode == :closed_form_only
end

# ---------------------------------------------------------------------------
# Test 4: mixed (some closed-form, some numeric) → :hybrid mode
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: hybrid mode" begin
    # σ_η is closed-form; a, σ_obs are numeric
    dm = _simple_normal_re_dm()
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=2
    ))
    notes = NoLimits.get_notes(res)
    @test notes.closed_form_mstep_used == true
    @test notes.closed_form_mstep_mode == :hybrid
end

# ---------------------------------------------------------------------------
# Test 5: opt-out disables closed-form
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: opt-out via empty re_cov_params" begin
    dm = _simple_normal_re_dm()
    # Pass empty NamedTuples to suppress auto-detection
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=2,
        re_cov_params=NamedTuple(),
        re_mean_params=NamedTuple(),
        resid_var_param=NamedTuple()
    ))
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.closed_form_mstep_used == false
    @test notes.closed_form_mstep_mode == :none
end

# ---------------------------------------------------------------------------
# Test 6: get_notes accessor returns correct structure
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: get_notes structure" begin
    dm = _simple_normal_re_dm()
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=2
    ))
    notes = NoLimits.get_notes(res)
    @test notes isa NamedTuple
    @test notes.closed_form_mstep_used isa Bool
    @test notes.closed_form_mstep_mode isa Symbol
    @test notes.closed_form_mstep_mode in (:none, :hybrid, :closed_form_only)
    @test hasproperty(notes, :closed_form_builtin_eligibility)
end

# ---------------------------------------------------------------------------
# Test 7: MvNormal RE uses closed-form
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: MvNormal RE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
            Ω = RealPSDMatrix([1.0 0.0; 0.0 1.0])
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t  = [0.0, 1.0, 0.0, 1.0],
        y  = [0.1, 0.2, -0.1, 0.0]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=2
    ))
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    # Ω (RE covariance) is closed-form; a and σ are numeric
    @test notes.closed_form_mstep_used == true
    @test notes.closed_form_mstep_mode == :hybrid
end

# ---------------------------------------------------------------------------
# Test 8: result converges faster with closed-form (sanity: no regression)
# ---------------------------------------------------------------------------
@testset "MCEM closed-form: result is a valid FitResult" begin
    dm = _simple_normal_re_dm()
    res_cf = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
        maxiters=5
    ))
    res_no = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
        maxiters=5,
        re_cov_params=NamedTuple(),
        re_mean_params=NamedTuple(),
        resid_var_param=NamedTuple()
    ))
    # Both should produce valid FitResults; objective is finite
    @test isfinite(NoLimits.get_objective(res_cf))
    @test isfinite(NoLimits.get_objective(res_no))
    # Closed-form result should report it was used
    @test NoLimits.get_closed_form_mstep_used(res_cf) == true
    @test NoLimits.get_closed_form_mstep_used(res_no) == false
end
