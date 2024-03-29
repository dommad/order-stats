"""Plotting results of all analyses"""
from typing import Tuple, List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from . import stat
from .utils import largest_factors
from .constants import TH_BETA, TH_MU
from .estimation import ParametersData
from .optimization_modes import LinearRegressionMode



class Plotting:
    """Plotting functionalities for the analysis of lower-order models"""

    def __init__(self, out_name) -> None:
        self.out_name = out_name

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial',
                             'xtick.labelsize': 10, 'ytick.labelsize': 10})


    def plot_mubeta(self, parameters_data: ParametersData, selected_hits: Tuple, **kwargs):
        
        expand = lambda df: (df.loc['location', :], df.loc['scale', :])

        plot_kwargs = {'marker': 'o', 'edgecolors': 'k', 'linewidth': 0.5}
        colors = ('#2D58B8', '#D65215')

        charges = parameters_data.available_charges

        fig, axs = plt.subplots(1, len(charges), figsize=(len(charges) * 4, 4))

        for idx, charge in enumerate(charges):
            this_charge_params_dict = parameters_data.output_dict[charge]

            for p_idx, item in enumerate(this_charge_params_dict.items()):
                p_estimator, param_df = item
                # TODO: allow selection of hits to plot
                param_df = param_df.loc[:, selected_hits]
                mu_vals, beta_vals = expand(param_df)
                axs[idx].scatter(mu_vals, beta_vals, color=colors[p_idx], label=p_estimator, **plot_kwargs)

                if kwargs.get('annotation'):
                    self.annotation(axs[idx], mu_vals, beta_vals, colors[p_idx])

                if kwargs.get('linear_regression'):
                    linreg = LinearRegressionMode(param_df).find_best_linear_regression()
                    self.add_linear_regression(axs[idx], mu_vals, linreg, color=colors[p_idx])


            axs[idx].set_xlabel(r"$\mu$")
            axs[idx].set_ylabel(r"$\beta$")
            axs[idx].set_title(f"charge {charge}")

        fig.tight_layout()
        fig.savefig(f"./{self.out_name}_mubeta_params_annot_{kwargs.get('annotation')}_lr_{kwargs.get('linear_regression')}.svg", dpi=600)

    
    @staticmethod
    def add_linear_regression(axes, xs, linreg, color):
        """
        Add fitted linear regression to the mu-beta plot and show the 
        starting parameters as an asterisk
        """
        x_range = np.array([min(TH_MU, min(xs)), max(xs)])
        axes.plot(x_range, x_range * linreg.slope + linreg.intercept, color=color)
        axes.scatter([TH_MU], [TH_BETA], marker='*', s=100, color='green')


    @staticmethod
    def annotation(axes, x_text, y_text, color):
        """Add text annotation with the hit rank"""
        offset = 2
        for idx, pair in enumerate(zip(x_text, y_text)):
            axes.annotate(idx + offset, (pair[0], pair[1]-0.0002), color=color)

    
    @staticmethod
    def add_axis_labels(axs, n_col, n_row, mode='density'):

        if mode == 'density':
            ylab = 'density'
            xlab = 'TEV'
        elif mode == 'PP':
            ylab = 'empirical CDF'
            xlab = 'theoretical CDF'

        for idx in range(n_col * n_row):
            if idx % n_col == 0:
                axs[divmod(idx, n_col)].set_ylabel(ylab)

            if divmod(idx, n_col)[0] == n_row-1:
                axs[divmod(idx, n_col)].set_xlabel(xlab)

    @staticmethod
    def get_number_lower_hits(param_dict):

        first_key = list(param_dict.keys())[0]
        sample_df = param_dict[first_key]
        if isinstance(sample_df, pd.DataFrame):
            return param_dict[first_key].shape[1]
        else:
            raise TypeError(f"The value in parameters dictionary should be pd.DataFrame, but it is {sample_df}")

    @staticmethod
    def get_optimal_subplot_numbers(num_entries):
        return largest_factors(num_entries)


    def plot_lower_models(self, df, score, parameters_data: ParametersData):
        """Plot density and PP plots for models of lower order TEV distributions"""

        charges = parameters_data.available_charges

        for charge in charges:
            this_charge_params_dict = parameters_data.output_dict[charge]
            this_charge_df = df[df['charge'] == charge]

            num_lower_hits = self.get_number_lower_hits(this_charge_params_dict)
            n_row, n_col = self.get_optimal_subplot_numbers(num_lower_hits)
    
            fig, axes = plt.subplots(n_row, n_col, figsize=(n_row*3, n_col*3), constrained_layout=True)
            idx_combinations = [(i, j) for i in range(n_row) for j in range(n_col)]

            for hit_idx in range(num_lower_hits):
                hit_rank = hit_idx + 2 # we skip hit number 1 as it's a mixture
                this_hit_scores = this_charge_df.loc[this_charge_df['hit_rank'] == hit_rank, score].values
        
                self.add_lower_model_plot(axes[idx_combinations[hit_idx]], this_hit_scores, this_charge_params_dict, hit_rank)
            
            self.add_axis_labels(axes, n_col, n_row, mode='density')
            fig.savefig(f"./{self.out_name}_lower_models_charge_{charge}.svg", dpi=600, bbox_inches="tight")
            

    @staticmethod
    def add_pp_plot(axs, cur_tevs, parameters_dict, hit_rank):
        """Add P-P plot of the model with rank equal to "hit_rank" to plt.subplots"""

        mle_pars = parameters_dict['mle'][0].loc[:, hit_rank]
        mm_pars = parameters_dict['mm'][0].loc[:, hit_rank]

        mm_pp = stat.TEVDistribution().cdf_asymptotic(cur_tevs, mm_pars[0], mm_pars[1], hit_rank)
        mle_pp = stat.TEVDistribution().cdf_asymptotic(cur_tevs, mle_pars[0], mle_pars[1], hit_rank)
        emp_pp = np.arange(1, len(cur_tevs) + 1) / len(cur_tevs)

        axs.scatter(mle_pp, emp_pp, color='#D65215', s=1)
        axs.scatter(mm_pp, emp_pp, color='#2CB199', s=1)
        axs.plot([0,1], [0,1], color='k')


    @staticmethod
    def add_lower_model_plot(axs, scores, estimator_params: dict, hit_rank):
        """Plotting KDE for all estimation methods for given hit_rank"""
            


        colors = ('#2D58B8', '#D65215', '#2CB199')

        kde_xs, kde_ys_observed = FFTKDE(bw=0.0005, kernel='gaussian').fit(scores).evaluate(2**8)
        axs.plot(kde_xs, kde_ys_observed, color='grey')
        

        if len(kde_ys_observed) == 0:
            return 0
        
        for idx, (p_estimator, param_df) in enumerate(estimator_params.items()):
            parameters = param_df.loc[:, hit_rank].values
            mu, beta = parameters
            pdf_vals = stat.TEVDistribution().pdf(kde_xs, mu, beta, hit_rank)
            axs.plot(kde_xs, pdf_vals, color=colors[idx], label=p_estimator)

        axs.set_ylim(0,)
        axs.set_title(f"hit {hit_rank}", fontsize=10)

    @staticmethod
    def get_rough_pi0_estimate(scores, mu, beta, hit_rank):

        pi0 = len(scores[scores < 0.2]) / len(scores)
        xs, kde_observed = FFTKDE(bw=0.01, kernel='gaussian').fit(scores).evaluate(2**8)
        pdf_fitted = stat.TEVDistribution().pdf(xs, mu, beta, hit_rank=hit_rank)

        return pi0, xs, kde_observed, pdf_fitted



    def plot_top_model_with_pi0(self, df, optimal_parameters: dict, score):
        """find pi0 estimates for plotting the final models"""

        num_charges = len(optimal_parameters)
        colors = ('#2CB199', '#2CB199', '#D65215')
        fig, axs = plt.subplots(1, num_charges, figsize=(5 * num_charges, 5))


        for idx, charge in enumerate(optimal_parameters):
            
            top_scores = df[(df['charge'] == charge) & (df['hit_rank'] == 1) ][score].values
            estimation_setting, (mu, beta) = optimal_parameters[charge]

            # get a rough approximation of pi0 for plotting only
            # TODO: to be abstracted out

            pi0, xs, kde_observed, pdf_fitted = self.get_rough_pi0_estimate(top_scores, mu, beta, 1) # we only work on top hits
           

            axs[idx].fill_between(xs, kde_observed, alpha=0.2, color=colors[0], label='observed')
            axs[idx].plot(xs, kde_observed, color=colors[1])
            axs[idx].plot(xs,  pi0 * pdf_fitted, color=colors[2], linestyle='-', label='fitted')
            axs[idx].set_xlim(0.0, 0.6)
            axs[idx].set_ylim(0,)
            axs[idx].set_xlabel("TEV")
            axs[idx].set_ylabel("density")

        fig.tight_layout()
        fig.savefig(f"./{self.out_name}_fitted_top_models.svg", dpi=600, bbox_inches="tight")

    ### plotting for BIC ###

    def plot_bic_diffs(self, bic_diffs, charge):
        """Plot BIC differences for TEV distributions for given charge state"""

        fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
        color = '#2D58B8'

        xs = np.arange(len(bic_diffs)) + 2 # we start from 2nd hit
        axs.scatter(xs, bic_diffs, color=color)
        axs.plot(xs, bic_diffs, color=color)

        axs.set_xlabel("hit_rank")
        axs.set_ylabel("relative BIC difference [%]")

        fig.savefig(f"./{self.out_name}_lower_models_BIC_charge_{charge}.svg", dpi=600, bbox_inches="tight")

    
    #### plotting for validation #####

   

        


class PlotValidation:

    @staticmethod
    def plot_fdp_fdr_cis(axs, fdps: List[Tuple]):
        """Plotting the FDR vs. estimated FDP + bootstrapped CIs"""

        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial',
                                'xtick.labelsize': 10, 'ytick.labelsize': 10})

        support = np.linspace(0.001, 0.1, 100) # TODO: abstract out
        means, upper_lims, lower_lims = list(np.array(x) for x in zip(*fdps))

        sns.lineplot(x=support, y=means, label='Mean', color='#2D58B8',ax=axs)
        sns.lineplot(x=[0, 0.1], y=[0, 0.1], linestyle='--', color='gray', ax=axs)
        axs.fill_between(support, upper_lims, lower_lims, color='royalblue', alpha=0.3, label='CI')

        axs.set_xlim(0, 0.1)
        axs.set_ylim(0, 0.1)
        # Customize the plot
        axs.set_xlabel('Estimated False Discovery Rate')
        axs.set_ylabel('False Discovery Proportion')
        plt.legend()


    @staticmethod
    def plot_tpr_fdr_cis(axs, tprs: List[Tuple]):
        """Plotting the FDR vs. estimated FDP + bootstrapped CIs"""

        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.family': 'Arial',
                                'xtick.labelsize': 10, 'ytick.labelsize': 10})

        support = np.linspace(0.001, 0.1, 100) # TODO: abstract out
        means, upper_lims, lower_lims = list(np.array(x) for x in zip(*tprs))

        sns.lineplot(x=support, y=means, label='Mean', color='#2D58B8', ax=axs)
        #sns.lineplot(x=[0, 0.1], y=[0, 0.1], linestyle='--', color='gray', ax=axs)
        axs.fill_between(support, upper_lims, lower_lims, color='royalblue', alpha=0.3, label='CI')

        # Customize the plot
        axs.set_ylabel('True Positive Rate')
        axs.set_xlabel('False Discovery Rate')
        plt.legend()



    def plot_validation_results(self, fdps, tprs):
        """Plot validation results"""

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        self.plot_fdp_fdr_cis(axs[0], fdps)
        self.plot_tpr_fdr_cis(axs[1], tprs)
        fig.tight_layout()

        fig.savefig(f"./fdp_tpr_fdr_validation.pdf", dpi=600, bbox_inches='tight')