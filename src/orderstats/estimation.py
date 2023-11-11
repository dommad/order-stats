"""Estimation of distribution parameters based on lower-order TEV distributions"""
from orderstats.utils import *
from orderstats import stat as of
from orderstats import parsers, plot
import scipy



class LowerOrderEstimation:
    
    def __init__(self, outname, bic_cutoff=0.17, lr_cutoff=(5,10)):
        self.params_est = []
        self.out = outname
        self.len_correct = 0
        self.bic_cutoff = bic_cutoff
        self.tevs = []
        self.lr_cutoff = lr_cutoff
        self.all_charges = [2,3,4]

    
    def run_estimation(self, input_path, pars_outname, engine='Tide', top_n=30):
        """Estimate parameters of top null model using lower order TEV distributions"""

        try:
            parser_instance = getattr(parsers, f"{engine}Parser")()
        except AttributeError:
            raise ValueError(f"Unsupported or invalid engine: {engine}")
        
        parser_instance = getattr(parsers, f"{engine}Parser")()

        lower_order_df = parser_instance.parse(input_path)
        lr_cutoff = (3, 9) #ranks of hits to include in linear regression fitting

        parameters_dict = ParameterEstimation().get_mle_mm_pars(lower_order_df, lr_cutoff)
        
        
        # estimate parameters for top models of all charges
        optimal_params = self.find_optimal_top_models(lower_order_df, parameters_dict)
        self.export_parameters_for_peptide_prophet(optimal_params, pars_outname)


        # plotting should be optional, so it will be at the end

        # plotting mu-beta combined and separate
        plot_object = plot.Plotting(self.out)
        plot_object.plot_mubeta(parameters_dict, methods=['mle', 'mm'])
        plot_object.plot_mubeta(parameters_dict, methods=['mle'], linear_regression=True, annotation=True)
        plot_object.plot_mubeta(parameters_dict, methods=['mm'], linear_regression=True, annotation=True)

        # plot MLE- and MM- lower-order models for each charge separately
        plot.Plotting().plot_mle_mm_lower_models(lower_order_df, parameters_dict)

        return optimal_params


    @staticmethod
    def export_parameters_for_peptide_prophet(params_est, params_out_name):
        """export params to txt for modified PeptideProphet (mean & std)"""

        params = pd.DataFrame.from_dict(params_est, orient='index', columns=['location', 'scale'])
        params['location'] += params['scale'] * np.euler_gamma # convert to mean
        params['scale'] = np.pi / np.sqrt(6) * params['scale'] # conver to std
        params.loc[1, :] = params.iloc[0, :] # add parameters for charge 1+ that we didn't consider

        # add parameters from highest present charge state to all missing higher charge states
        max_idx = max(params.index)
        len_missing = 10 - max_idx
        new_idx = range(max_idx + 1, 10 + 1)

        to_concat = pd.DataFrame(len_missing * (params.loc[max_idx, :],), index = new_idx, columns=['location', 'scale'])
        final_params = pd.concat([params, to_concat], axis=0, ignore_index=False)
        final_params.sort_index(inplace=True)

        final_params.to_csv(f"{params_out_name}.txt", sep=" ", header=None, index=None)



    @staticmethod
    def get_linear_regression(param_df, lr_cutoff):
        """get linear regression parameters and mean beta for further processing"""

        columns_within_cutoff = param_df.columns.isin(range(*lr_cutoff))
        selected_df = param_df.loc[:, columns_within_cutoff]
        mu_values = selected_df.loc['location', :]
        beta_values = selected_df.loc['scale', :]
        linreg_results = st.linregress(mu_values, beta_values)

        return linreg_results


    @staticmethod
    def get_mean_beta(param_df, min_rank=8):
        """get linear regression parameters and mean beta for further processing"""
        columns_within_cutoff = param_df.columns >= min_rank
        selected_df = param_df.loc[:, columns_within_cutoff]
        mean_beta = np.mean(selected_df.loc['scale'])

        return mean_beta
    

    @staticmethod
    def find_peaks_and_dips(axes, kde):
        """
        Find peaks in TEV data.

        Parameters:
        - data (array-like): TEV data.

        Returns:
        - indices (array): Indices of peaks in the data.
        """

        # peaks = []
        dips = []
        for i in range(1, len(kde) - 1):
            # if kde[i] > kde[i - 1] and kde[i] > kde[i + 1]:
            #     peaks.append(i)
            if kde[i] < kde[i - 1] and kde[i] < kde[i + 1]:
                dips.append(i)

        if len(dips) == 0:
            return []
        else:
            return axes[dips] #, axes[dips]
    


    def find_main_dip(self, data):
        """
        Find the main dip in the mixture distribution separating two components
        """

        axes, kde = FFTKDE(bw=0.05, kernel='gaussian').fit(data).evaluate(2**8)
        dips = self.find_peaks_and_dips(axes, kde)

        if len(dips) == 0:
            main_dip = np.median(data)
        else:
            main_dip = dips[max(0, int(len(dips / 2)) - 1)]
        
        return main_dip


    def find_best_estimation_method(self, top_tevs, parameters_dict):
        """
        Selecting best from (MLE-LR, MLE-mean, MM-LR, MM-beta)
        optimized with respect to BIC
        """
        
        optimal_results = {}
        methods = ['mle', 'mm']

        # we need to roughly cut off the upper portion of the mixture
        main_dip_cutoff = self.find_main_dip(top_tevs)
        sel_tevs = top_tevs[top_tevs < main_dip_cutoff]

        for method in methods:
            param_df, linreg = parameters_dict[method]
            
            # get optimal parameters for linear regression mode
            optimal_params_lr_mode = self.find_optimal_pars(sel_tevs, order=0, linreg=linreg)

            # get optimal parameters for mean beta mode
            mean_beta = self.get_mean_beta(param_df, min_rank=8)
            optimal_params_mean_beta_mode = self.find_optimal_pars(sel_tevs, order=0, beta=mean_beta)

            optimal_results[(method, 'lr')] = optimal_params_lr_mode
            optimal_results[(method, 'mean-beta')] = optimal_params_mean_beta_mode

        best_key = min(optimal_results, key=lambda k: optimal_results[k][1])

        return optimal_results, best_key
                


    def find_optimal_top_models(self, df, parameters_dict, plot=False):
        """estimate parameters of top models using lower order distributions"""

        optimal_params = {}

        # mode_dict  = {0: "MLE LR", 1: "MLE mean", 2: "MM LR", 3: "MM mean"}

        for charge in parameters_dict:

            cur_parameters = parameters_dict[charge] # MLE and MM parameters for given charge

            # extract top hits for the selected charge state
            # TODO: remove the top section of the mixture for better fitting, separate function
            top_tevs = df[df['charge'] == charge]['tev']
            top_tevs = top_tevs[top_tevs > 0.01] # dommad: don't count scores at 0 and below

            optim_results, best_key = self.find_best_estimation_method(top_tevs, cur_parameters)
            cur_optimal_params = optim_results[best_key][0]
            optimal_params[charge] = cur_optimal_params
        
        # plotting - optional
        if plot:
            self.plot_top_model_with_pi0(top_tevs, optimal_params) # for plotting
        
        return optimal_params
    


    @staticmethod
    def find_optimal_pars(data, order, beta=0.02, linreg=None):
        """
        Maximum Likelihood Estimation for TEV distribution
        """

        def get_log_likelihood(params, data, order, beta=None, linreg=None):
            """
            Optimizing BIC for the lower section of the distribution
            is equivalent to optimizing log-likelihood
            """
            mu = max(params[0], 1e-6)  # Extract mu from the parameters
            if linreg:
                beta = mu * linreg.slope + linreg.intercept

            logged_params = np.log(np.array([mu, beta]))

            log_like = of.AsymptoticGumbelMLE(data, order).get_log_likelihood(logged_params)
            return log_like

        
        initial_guess = np.mean(data)
        bounds = [(0.1 * initial_guess, 1.5 * initial_guess)]
        objective_function = lambda a, b, c, d, e: get_log_likelihood(a, b, c, d, e)

        results = scipy.optimize.minimize(
            fun = objective_function,
            x0 = np.array([initial_guess]),
            #x0 = initial_guess,
            args=(data, order, beta, linreg,),
            method='L-BFGS-B',
            bounds=bounds   
            )
        
        opt_mu = results.x[0]
        
        if linreg:
            opt_beta = opt_mu * linreg.slope + linreg.intercept
        else:
            opt_beta = beta
        
        return (opt_mu, opt_beta), results.fun
    


    def plot_top_model_with_pi0(self, data, pars, plot=True):
        """find pi0 estimates for plotting the final models"""

        # TODO: now pars is a dict: charge: (mu, beta)
        mu, beta = pars
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        kde = kde/auc(axes, kde)
        trunk = len(axes[axes < self.bic_cutoff])
        theory = of.TEVDistribution().pdf(axes, mu, beta, 0)
        err = 1000
        best_pi = 0

        for pi_0 in np.linspace(0, 1, 500):
            new_err = abs(auc(axes[:trunk], kde[:trunk]) - auc(axes[:trunk], pi_0*theory[:trunk]))
            if new_err < err:
                best_pi = pi_0
                err = new_err

        num_charges = len(pars)
        fig, axs = plt.subplots(1, num_charges, figsize=(2 * num_charges, 2))

        if plot:
            axs.fill_between(axes, kde, alpha=0.2, color='#2CB199')
            axs.plot(axes, kde, color='#2CB199')
            axs.plot(axes, best_pi*theory, color='#D65215', linestyle='-')
            axs.set_xlim(0.0, 0.6)
            axs.set_ylim(0,20)
            axs.set_xlabel("TEV")
            axs.set_ylabel("density")

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_alt_top_models.png", dpi=600, bbox_inches="tight")

    
    def get_bic_for_lower_models(self, lower_order_df, parameters_dict, charge, plot=False):
        """Calculate BIC for the lower-order models"""
        hit_ranks = set(lower_order_df['hit_rank'])
        hit_ranks.discard(1)

        bic_diffs = []
        charge_df = lower_order_df[lower_order_df['charge'] == charge].copy()
        charge_parameters = parameters_dict[charge]

        for hit_rank in hit_ranks:
            cur_tevs = charge_df[charge_df['hit_rank'] == hit_rank]['tev']
            cur_bic_diff = self.calculate_mle_mm_bic_diff(cur_tevs, hit_rank, charge_parameters)
            # cur_bic = self.compare_density_auc(self.tevs[idx][:,hit], mle_par, hit, idx)
            bic_diffs.append(cur_bic_diff)

            if plot:
                self.plot_bic_diffs(bic_diffs, charge)


    def calculate_mle_mm_bic_diff(self, tevs, hit_rank, parameters_dict, k=2):
        """
        Calculate difference between BIC for MLE and MM models
        for hit_rank=1
        """

        def get_bic(tevs, hit_rank, params):
            log_likelihood = of.AsymptoticGumbelMLE(tevs, hit_rank).get_log_likelihood(np.log(params))
            bic = k * np.log(len(tevs)) - 2 * log_likelihood
            return bic
        
        tevs = tevs[tevs < self.bic_cutoff]
        mle_params = parameters_dict['mle'][0].loc[:, hit_rank]
        mm_params = parameters_dict['mm'][0].loc[:, hit_rank]

        mle_bic = get_bic(tevs, hit_rank, mle_params)
        mm_bic = get_bic(tevs, hit_rank, mm_params)
        bic_diff = abs(mle_bic - mm_bic) / mle_bic

        return bic_diff

    
    def compare_density_auc(self, tevs, parameters_dict, hit_rank):
        """Compare densities of the observed TEV distribution and the model"""
        
        mu, beta = parameters_dict['mle'][0].loc[:, hit_rank]
        xs, kde_observed = FFTKDE(bw=0.0005, kernel='gaussian').fit(tevs).evaluate(2**8)
        auc_observed = auc(xs, kde_observed)
        density_model = of.TEVDistribution().pdf(xs, mu, beta, hit_rank)
        auc_model = auc(xs, density_model)

        return abs(auc_model - auc_observed) / auc_observed
    


class ParameterEstimation:
    
    def __init__(self) -> None:
        pass


    def get_linreg(self, param_df, lr_cutoff):
        """Fit linear regression to mu and beta values 
        from lower orders and return the results"""

        columns_within_cutoff = param_df.columns.isin(range(*lr_cutoff))
        selected_df = param_df.loc[:, columns_within_cutoff].copy()
        mu_values = selected_df.loc['location', :]
        beta_values = selected_df.loc['scale', :]
        linreg_results = st.linregress(mu_values, beta_values)

        return linreg_results


    @staticmethod
    def find_pars(df, method='mle'):
        """Get method of moments parameters"""

        available_hits = set(df['hit_rank'])
        available_hits.discard(1) # skip first hit since it's a mixture
        params = {}

        for hit in available_hits:
            df_hit = df[df['hit_rank'] == hit]
            hit_tevs = df_hit[df_hit['tev'] > 0.01]['tev'] # dommad
            if method == 'mm':
                hit_params = of.MethodOfMoments().estimate_parameters(hit_tevs, hit)
            elif method == 'mle':
                hit_params = of.AsymptoticGumbelMLE(hit_tevs, hit).run_mle()
            else:
                print("the method is incorrect!")

            params[hit] = hit_params
    
        params_df = pd.DataFrame.from_records(params, index=['location', 'scale'])
        return params_df

    
    def get_mle_mm_pars(self, df, lr_cutoff):
        """calculate MLE and MM parameters using the data"""
        available_charges = set(df['charge'])
        methods = ['mle', 'mm']
        output_dict = {}

        for charge in available_charges:
            charge_dict = {}
            sel_df = df[df['charge'] == charge].copy()

            for method in methods:
                method_params = self.find_pars(sel_df, method=method)
                method_lr = self.get_linreg(method_params, lr_cutoff)
                charge_dict[method] = (method_params, method_lr)

            output_dict[charge] = charge_dict

        return output_dict



class PiZeroEstimator:
    """estimate pi0 for given set of p-values"""

    def __init__(self):
        pass

    @staticmethod
    def get_pi0_b(pvs, b_val):
        """calculate pi0 estimate for given b value"""
        i = 1
        condition = False

        while condition is False:
            t_i = (i-1)/b_val
            t_iplus = i/b_val
            ns_i = len(pvs[(pvs < t_iplus) & (pvs >= t_i)])
            nb_i = len(pvs[pvs >= t_i])
            condition = bool(ns_i <= nb_i/(b_val - i + 1))
            i += 1

        i -= 1

        summand = 0
        for j in range(i-1, b_val+1):
            t_j = (j-1)/b_val
            summand += len(pvs[pvs >= t_j])/((1-t_j)*len(pvs))

        pi_0 = 1/(b_val - i + 2)*summand

        return pi_0

    def get_all_pi0s(self, pvs):
        """calculate pi0 for each b value"""
        # B is from I = {5, 10, 20, 50, 100}

        pi0s = []
        b_set = [5, 10, 20, 50, 100]

        for b_val in b_set:
            pi0s.append(self.get_pi0_b(pvs, b_val))
        
        return pi0s


    def get_boostrap_pi0s(self, pvs, no_reps, b_val):

        pi0_estimates = np.zeros(no_reps)

        for rep in range(no_reps):
            random.seed()
            new_pvs = np.array(random.choices(pvs, k=len(pvs)))
            pi0_estimates[rep] = self.get_pi0_b(new_pvs, b_val)

        return pi0_estimates

    @staticmethod
    def get_mse(pi0_bootstrap, pi0_true):
        """Calculates MSE for given set of p-values and true pi0 value"""
        summand = 0
        for i in range(len(pi0_bootstrap)):
            summand += pow(pi0_bootstrap[i] - pi0_true, 2)
        
        return summand/len(pi0_bootstrap)


    def find_optimal_pi0(self, pvs, n_reps):
        """Find the optimal pi0 according to Jiang and Doerge (2008)"""
        # compute pi0 for each B
        pi0_estimates = self.get_all_pi0s(pvs)
        pi0_ave = np.mean(pi0_estimates)
        b_set = [5, 10, 20, 50, 100]

        # compute MSE for each pi0 estimate
        mses = []

        for pi0_estim, b_val in zip(pi0_estimates, b_set):
            bootstraps = self.get_boostrap_pi0s(pvs, n_reps, b_val)
            mses.append(self.get_mse(bootstraps, pi0_ave))

        optimal_idx = mses.index(sorted(mses)[0])
        print(mses)
        print(pi0_estimates)
        print(optimal_idx)
        return pi0_estimates[optimal_idx]
