import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.stats import qmc
from numba import njit
from joblib import Parallel, delayed
import jax
import jax.numpy as jnp
from jax import grad, jit

# _________________________________
# Definging Steady-State Functions
# _________________________________

@njit
def steady_state_equation(x, beta, gamma):
    # steady-state equation where f(x) = 0
    return 1 - beta * x + (gamma * x**2) / (1 + x**2)

@njit
def df_dx(x, beta, gamma):
    #Derivative of steady state df/dx = 0
    numerator = -beta + gamma * (2 * x * (1 + x**2) - 2 * x**3) / (1 + x**2)**2
    return numerator

# _______________________________
# Root-Finding Function
# _______________________________

# I found a great source that showed me how to do this using the numba and scipi library so I decided to just repurpose it

def find_steady_states(beta, gamma, x_min=0.01, x_max=100.0, num_points=1000):
    
    # Newtons method Root Finder
    x_initial_guesses = np.linspace(x_min, x_max, num_points)
    roots = []
    for x0 in x_initial_guesses:
        try:
            # Scifi-compiled functions
            root = newton(
                steady_state_equation,
                x0,
                fprime=df_dx,
                args=(beta, gamma),
                maxiter=50,
                tol=1e-8
            )
            if x_min <= root <= x_max:
                # Checking for uniqueness within a tolerance to exclude dupilicates 
                if not any(np.isclose(root, existing_root, atol=1e-6) for existing_root in roots):
                    roots.append(root)
        except (RuntimeError, OverflowError):
            continue
    return np.array(roots)

# ______________________________________________
# Analyze Parameter Space with Parallelization
# _____________________________________________


def analyze_single_parameter(beta, gamma):
    # Analyzes a single (beta, gamma) parameter pair to find steady states and their stability.
    roots = find_steady_states(beta, gamma)
    num_roots = len(roots)
    stable = 0
    for root in roots:
        stability = df_dx(root, beta, gamma)
        if stability < 0:
            stable += 1
    return (beta, gamma, num_roots, stable)

# I wanted to try out parallel computing and this is my attempt, JobLib basically does most of the work for you once you figure out how to enter paramtetsr
def analyze_parameter_space_parallel(beta_samples, gamma_samples, n_jobs=-1):
    # Parallelizing the analysis of parameter space using Joblib

    # This is kind of complicated and honestly I don't entirely know what's going on but here is my best explanation

    # We are running analyze_single_parameter on each (beta, gamma) pair in parallel
    # the delayed() sets up each function call as a task for parallel execution
    # zip(beta_samples, gamma_samples) pairs each beta with its corresponding gamma value
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(analyze_single_parameter)(beta, gamma) for beta, gamma in zip(beta_samples, gamma_samples))
    beta_vals, gamma_vals, num_steady_states, num_stable_states = zip(*results)
    return (np.array(beta_vals), np.array(gamma_vals), np.array(num_steady_states), np.array(num_stable_states))

# _______________________________
# Plotting Steady State Functions
# _______________________________

# Note: I decided to just save the files into the directory for ease of seeing 
def plot_scatter(beta_vals, gamma_vals, num_steady_states, title, cmap='viridis', filename=None):
    # Parameter space colored by the number of steady states.
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(beta_vals, gamma_vals, c=num_steady_states, cmap=cmap, marker='o', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Number of Steady States')
    plt.xlabel('Beta (β)')
    plt.ylabel('Gamma (γ)')
    plt.title(title)
    if filename:
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_steady_state_function(beta, gamma, x_min=0.01, x_max=100.0, num_points=1000, filename=None):
    # Plots the steady-state equation at f(x)=0
    x = np.linspace(x_min, x_max, num_points)
    f = steady_state_equation(x, beta, gamma)
    plt.figure(figsize=(10, 6))
    plt.plot(x, f, label='f(x)')
    plt.axhline(0, color='red', linestyle='--', label='f(x) = 0')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Steady-State Equation for β={beta:.4f}, γ={gamma:.4f}')
    plt.legend()
    if filename:
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_potential(x_array, U, title='Potential Function U(x)', filename=None):
    # Plots the potential function U(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x_array, U)
    plt.xlabel('x')
    plt.ylabel('U(x)')
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()
    else:
        plt.show()

# _______________________________________
# Langevin Equation Simulation Functions
# _______________________________________

# Creating a  trajectory of the Langevin equation using Euler-Maruyama method
def simulate_langevin(beta, gamma, x0, t_span, dt, sigma):
    # Calculate the total number of time steps needed
    num_steps = int((t_span[1] - t_span[0]) / dt)
    t = np.linspace(t_span[0], t_span[1], num_steps)
    x = np.zeros(num_steps)
    x[0] = x0
    for i in range(1, num_steps):
        # Calculating the deterministic term
        drift = steady_state_equation(x[i-1], beta, gamma)
        # Applying stochastic portion of model
        x[i] = x[i-1] + drift * dt + sigma * np.sqrt(dt) * np.random.randn()
        # Ensuring x stays positive
        if x[i] < 0:
            x[i] = 0
    return t, x

# Distribution of outocmes 
def langevin_distribuition(beta, gamma, x0, t_span, dt, sigma, n_simulations):
    final_x = []
    for i in range(n_simulations):
        _, x = simulate_langevin(beta, gamma, x0, t_span, dt, sigma)
        final_x.append(x[-1])
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_simulations} simulations.")
    return np.array(final_x)

# ______________________________
# Fokker-Planck Equation Solver
# _____________________________

def solve_fokker_planck(beta, gamma, x_array, sigma):
   # intitalizing the step size
    dx = x_array[1] - x_array[0]
    # setting up the diffusion coefficient
    D = sigma**2 / 2
    # Computing the drift term
    drift = 1 - beta * x_array + (gamma * x_array**2) / (1 + x_array**2)

    # intiaiilizing the matrices that we will use to solve this system of equations
    N = len(x_array)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Setting up the itnterior points using central differences
    for i in range(1, N - 1):
        A[i, i - 1] = D / dx**2 - drift[i] / (2 * dx)
        A[i, i] = -2 * D / dx**2
        A[i, i + 1] = D / dx**2 + drift[i] / (2 * dx)
        b[i] = 0

    # Need to apply boundary conditions to ensure zero flux at the boundaries (Neumann boundary conditions)
    # Left boundary (i = 0) set zero flux
    A[0, 0] = -2 * D / dx**2
    A[0, 1] = 2 * D / dx**2
    b[0] = 0 # zero flux at the left edge

    # Right boundary (i = N-1) to set zero-flux
    A[N - 1, N - 2] = 2 * D / dx**2
    A[N - 1, N - 1] = -2 * D / dx**2
    b[N - 1] = 0 # zero flux at the right edge

    # Solving linear system with error handling (There was a lot of errors initailzy)
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print("Linear algebra error:", e)
        return np.zeros_like(x_array)

    # Ensures non-negative probabilities
    P[P < 0] = 0

    # This was mostly for debugging but i'll keep it just in case 
    if np.any(np.isnan(P)) or np.any(np.isinf(P)):
        print("NaNs or Infinities Found")
        return np.zeros_like(x_array)

    # Normalizing the distribution
    integral = trapezoid(P, x_array)
    if integral > 0:
        P /= integral
    else:
        print("P value is not valid")
        P = np.zeros_like(x_array)

    return P

# ________________________________
# Optimization Function Using JAX
# ________________________________

# Optimizing beta and gama values to minimize variance of distribution
def optimize_parameters_jax(initial_params, x_array, sigma, bounds):
    # turning initail params into arrays 
    initial_params = jnp.array(initial_params)
    lower_bounds = jnp.array([bounds[0][0], bounds[1][0]])
    upper_bounds = jnp.array([bounds[0][1], bounds[1][1]])

    # Objective function we want to minimize
    @jit
    def objective(params):
        beta, gamma = params
        drift = 1 - beta * x_array + (gamma * x_array**2) / (1 + x_array**2)
        U = -jnp.cumsum(drift) * (x_array[1] - x_array[0])
        U -= jnp.min(U)
        D = sigma**2 / 2
        P = jnp.exp(-U / D)
        integral = jnp.trapezoid(P, x_array)  
        P = jnp.where(integral > 0, P / integral, jnp.zeros_like(P))
        mean = jnp.trapezoid(P * x_array, x_array)  
        mean_sq = jnp.trapezoid(P * x_array**2, x_array)
        variance = mean_sq - mean**2
        return variance

    # Gradient of the objective function
    grad_objective = jit(grad(objective))

    # Optimization loop
    params = initial_params
    learning_rate = 0.01
    for i in range(1000):
        grads = grad_objective(params)
        params = params - learning_rate * grads
        # Apply bounds
        params = jnp.clip(params, lower_bounds, upper_bounds)
        if i % 100 == 0:
            var = objective(params)
            print(f"Iteration {i}: Variance = {var:.6f}, Beta = {params[0]:.4f}, Gamma = {params[1]:.4f}")

    optimized_params = params
    optimized_variance = objective(optimized_params)
    return {
        'beta': optimized_params[0].item(),
        'gamma': optimized_params[1].item(),
        'variance': optimized_variance.item()
    }

# _______________
# Main Execution
# ________________

if __name__ == "__main__":
    

    # Analyzing Parameter Space with Parallelization
    # Defining parameter ranges
    num_samples = 5000  
    beta_range = (0.1, 20)
    gamma_range = (0.1, 50)
    np.random.seed(42) 

    # I found that parameter samples using Latin Hypercube Sampling gives thoes most optimal coverge
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=num_samples)

    # Scaling the samples
    l_bounds = [beta_range[0], gamma_range[0]]
    u_bounds = [beta_range[1], gamma_range[1]]
    beta_gamma_samples = qmc.scale(sample, l_bounds, u_bounds)
    beta_samples = beta_gamma_samples[:, 0]
    gamma_samples = beta_gamma_samples[:, 1]

    # Verify the samples
    print(f"Beta samples range from {beta_samples.min():.4f} to {beta_samples.max():.4f}")
    print(f"Gamma samples range from {gamma_samples.min():.4f} to {gamma_samples.max():.4f}")

    print("\nStarting parameter space analysis with sampling...")
    beta_vals, gamma_vals, num_steady_states, num_stable_states = analyze_parameter_space_parallel(beta_samples, gamma_samples)

    # Print the stats we calculated into terminal 
    print("\n--- Parameter Space Analysis Summary ---")
    print(f"Total Samples Analyzed: {num_samples}")
    print(f"Steady States - Min: {num_steady_states.min()}, Max: {num_steady_states.max()}, Mean: {num_steady_states.mean():.2f}")
    print(f"Stable Steady States - Min: {num_stable_states.min()}, Max: {num_stable_states.max()}, Mean: {num_stable_states.mean():.2f}")
    print("----------------------------------------")

    # Plotting the results
    plot_scatter(
        beta_vals,
        gamma_vals,
        num_steady_states,
        title='Parameter Space Sampling - Number of Steady States',
        cmap='viridis',
        filename='parameter_space_sampling'
    )
    print("Scatter plot saved as 'parameter_space_sampling.png'.")



    # Identifying and testing hte parameters with multiple staeady states 
    # Identify indices where multiple steady states are present
    multistable_indices = np.where(num_steady_states > 1)[0]
    if multistable_indices.size > 0:
        # Select a parameter set with the maximum number of steady states
        max_roots = np.max(num_steady_states[multistable_indices])
        candidates = multistable_indices[num_steady_states[multistable_indices] == max_roots]
        selected_idx = candidates[0]
        beta_multistable = beta_vals[selected_idx]
        gamma_multistable = gamma_vals[selected_idx]
        print(f"\nParameters with multiple steady states detected: β={beta_multistable:.4f}, γ={gamma_multistable:.4f}")
    else:
        print("\nNo parameters with multiple steady states detected.")
        beta_multistable = 2.5  # Default values if none detected
        gamma_multistable = 15.0

    # Testing the selected parameters
    print(f"\nTesting parameters: β={beta_multistable:.4f}, γ={gamma_multistable:.4f}")
    roots = find_steady_states(beta_multistable, gamma_multistable)
    stabilities = [df_dx(root, beta_multistable, gamma_multistable) for root in roots]
    num_stable = sum(1 for s in stabilities if s < 0)
    print(f"Steady states: {roots}")
    print(f"Stabilities: {stabilities}")
    print(f"Number of stable states: {num_stable}")

    # Printing all of the the steady states and stabilities we calujcated 
    print("\n--- Steady States and Their Stabilities ---")
    for i, (state, stability) in enumerate(zip(roots, stabilities), 1):
        status = "Stable" if stability < 0 else "Unstable"
        print(f"Steady State {i}: x = {state:.6f}, Stability = {stability:.6f} ({status})")
    print("--------------------------------------------")

    # Graph of the steady-state equation
    plot_steady_state_function(
        beta_multistable,
        gamma_multistable,
        x_min=0.01,
        x_max=100.0,
        num_points=1000,
        filename=f'steady_state_function_beta_{beta_multistable:.4f}_gamma_{gamma_multistable:.4f}'
    )
    print(f"Steady-state equation plot saved as 'steady_state_function_beta_{beta_multistable:.4f}_gamma_{gamma_multistable:.4f}.png'.")

    

    # Langevin Equation Simulations
    print("\nSimulating Langevin equation to generate steady-state distributions...")

    # Parameters for simulation
    x0 = 0.1  # Initial condition
    t_span = (0, 200)  # Time span
    dt = 0.01  # Time step
    sigma = 0.5  # Noise intensity
    n_simulations = 1000  # Number of simulations

    # Run ensemble simulations
    final_x = langevin_distribuition(
        beta_multistable,
        gamma_multistable,
        x0,
        t_span,
        dt,
        sigma,
        n_simulations
    )

    # Creatung histogram of final x values
    plt.figure(figsize=(10, 6))
    plt.hist(final_x, bins=100, density=True, alpha=0.6, label='Langevin Simulations')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Steady-State Distribution from Langevin Simulations (β={beta_multistable:.4f}, γ={gamma_multistable:.4f})')
    plt.legend()
    plt.savefig('langevin_steady_state_distribution.png', dpi=300)
    plt.close()
    print("Histogram of Langevin simulations saved as 'langevin_steady_state_distribution.png'.")

    
    
    
    # Solving the Fokker-Planck Equation
    print("\nSolving the Fokker-Planck equation numerically...")


    x_array_fp = np.linspace(0.01, 100.0, 5000)  
    P_fp = solve_fokker_planck(beta_multistable, gamma_multistable, x_array_fp, sigma)

    # Errot hanlding to make sure that the P_fp contains valid values
    if np.all(P_fp == 0):
        print("Fokker-Planck solution contains all zeros. Check the solver parameters and boundary conditions.")
    else:
        # Plotting the Fokker-Planck solution
        plt.figure(figsize=(10, 6))
        plt.plot(x_array_fp, P_fp, 'r-', label='Fokker-Planck Solution')
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.title(f'Steady-State Distribution from Fokker-Planck Equation (β={beta_multistable:.4f}, γ={gamma_multistable:.4f})')
        plt.legend()
        plt.savefig('fokker_planck_steady_state_distribution.png', dpi=300)
        plt.close()
        print("Fokker-Planck solution plot saved as 'fokker_planck_steady_state_distribution.png'.")

    
    print("\nComparing distributions from Langevin simulations and Fokker-Planck solution...")

    plt.figure(figsize=(10, 6))
    plt.hist(final_x, bins=100, density=True, alpha=0.6, label='Langevin Simulations')
    if not np.all(P_fp == 0):
        plt.plot(x_array_fp, P_fp, 'r-', label='Fokker-Planck Solution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Comparison of Steady-State Distributions (β={beta_multistable:.4f}, γ={gamma_multistable:.4f})')
    plt.legend()
    plt.savefig('comparison_steady_state_distribution.png', dpi=300)
    plt.close()
    print("Comparison plot saved as 'comparison_steady_state_distribution.png'.")


    # Optimization portion
    print("\nStarting optimization to minimize variance of steady-state distribution using JAX...")

    # Define x_array for computations
    x_array_jax = jnp.linspace(0.01, 100.0, 5000)  # Increased grid resolution

    # Initial guess and bounds for optimization
    initial_guess = [2.5, 15.0]  # Starting with default values
    bounds = [(0.1, 20), (0.1, 50)]  # Same as parameter ranges

    # Optimizzatino parameters
    opt_result = optimize_parameters_jax(initial_guess, x_array_jax, sigma, bounds)

    beta_opt = opt_result['beta']
    gamma_opt = opt_result['gamma']
    variance_opt = opt_result['variance']

    print(f"\nOptimized Parameters: β = {beta_opt:.4f}, γ = {gamma_opt:.4f}")
    print(f"Optimized Variance: {variance_opt:.6f}")

    # Computing the optimized steady-state distribution
    x_array_opt = np.linspace(0.01, 100.0, 5000)
    drift_opt = 1 - beta_opt * x_array_opt + (gamma_opt * x_array_opt**2) / (1 + x_array_opt**2)
    U_opt = -cumulative_trapezoid(drift_opt, x_array_opt, initial=0)
    U_opt -= np.min(U_opt)  # Normalize potential
    D_opt = sigma**2 / 2
    P_opt = np.exp(-U_opt / D_opt)
    integral_opt = trapezoid(P_opt, x_array_opt)
    if integral_opt > 0:
        P_opt /= integral_opt
    else:
        print("Optimized steady-state distribution contains all zeros. Check the optimization process.")

    # Graph of hte optimized steady-state distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x_array_opt, P_opt, 'b-', label='Optimized Steady-State Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Optimized Steady-State Distribution (β={beta_opt:.4f}, γ={gamma_opt:.4f}, σ={sigma})')
    plt.legend()
    plt.savefig('optimized_steady_state_distribution.png', dpi=300)
    plt.close()
    print("Optimized steady-state distribution plot saved as 'optimized_steady_state_distribution.png'.")

    


    # Plotting noise data
    print("\nInvestigating how optimized parameters change with different noise levels...")

    sigma_values = [0.1, 0.5, 1.0, 2.0]
    optimized_params_list = []

    for sigma_val in sigma_values:
        print(f"\nOptimizing for σ = {sigma_val}")
        opt_result = optimize_parameters_jax(initial_guess, x_array_jax, sigma_val, bounds)
        beta_opt_val = opt_result['beta']
        gamma_opt_val = opt_result['gamma']
        variance_opt_val = opt_result['variance']
        optimized_params_list.append((sigma_val, beta_opt_val, gamma_opt_val, variance_opt_val))
        print(f"Optimized Parameters: β = {beta_opt_val:.4f}, γ = {gamma_opt_val:.4f}, Variance: {variance_opt_val:.6f}")

    # Plotting the optimized beta and gamma we found versus sigma
    sigma_vals, betas_opt, gammas_opt, variances_opt = zip(*optimized_params_list)
    plt.figure(figsize=(10, 6))
    plt.plot(sigma_vals, betas_opt, 'o-', label='Optimized β')
    plt.plot(sigma_vals, gammas_opt, 's-', label='Optimized γ')
    plt.xlabel('Noise Intensity σ')
    plt.ylabel('Optimized Parameters')
    plt.title('Optimized Parameters vs. Noise Intens    ity')
    plt.legend()
    plt.savefig('optimized_parameters_vs_noise.png', dpi=300)
    plt.close()
    print("Plot of optimized parameters vs. noise intensity saved as 'optimized_parameters_vs_noise.png'.")

