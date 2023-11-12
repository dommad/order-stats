import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from . import stat as of


TH_N0 = 1000.
TH_MU = 0.02 * np.log(TH_N0)
TH_BETA = 0.02

class Plotting:
    """Plotting functionalities for the analysis of lower-order models"""

    def __init__(self, out_name) -> None:
        self.out_name = out_name

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 13, 'font.family': 'Helvetica'})




    def plot_mubeta(self, parameters_dict, methods = ['mle', 'mm'], **kwargs):
        
        expand = lambda df: (df.loc['location', :], df.loc['scale', :])

        plot_kwargs = {'marker': 'o', 'edgecolors': 'k', 'linewidth': 0.5}
        colors = {'mle': '#2D58B8', 'mm': '#D65215'}

        charges = parameters_dict.keys()
        num_charges = len(charges)
        fig, axs = plt.subplots(1, num_charges, figsize=(num_charges * 4, 4))

        for idx, charge in enumerate(charges):
            cur_charge = parameters_dict[charge]

            for method in methods:
                xs, ys = expand(cur_charge[method][0])
                axs[idx].scatter(xs, ys, color=colors[method], **plot_kwargs)

                if kwargs.get('annotation'):
                    self.annotation(axs[idx], xs, ys, colors[method])

                if kwargs.get('linear_regression'):
                    linreg = cur_charge[method][1]
                    self.add_linear_regression(axs[idx], xs, linreg, color=colors[method])

            axs[idx].set_xlabel(r"$\mu$")
            axs[idx].set_ylabel(r"$\beta$")
            axs[idx].set_title(f"charge {charge}")

        fig.tight_layout()
        fig.savefig(f"./{self.out_name}_mubeta_params_annot_{kwargs.get('annotation')}_lr_{kwargs.get('linear_regression')}_{methods}.png", dpi=600)

    
    @staticmethod
    def add_linear_regression(axes, xs, linreg, color):
        """
        Add fitted linear regression to the mu-beta plot and show the 
        starting parameters as an asterisk
        """
        x_range = np.array([min(TH_MU, min(xs)), max(xs)])
        axes.plot(x_range, x_range * linreg.slope + linreg.intercept)
        axes.scatter([TH_MU], [TH_BETA], marker='*', s=20, color='orange')


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

        for idx in range(n_col*n_row):
            if idx % n_col == 0:
                axs[divmod(idx, n_col)].set_ylabel(ylab)

            if divmod(idx, n_col)[0] == n_row-1:
                axs[divmod(idx, n_col)].set_xlabel(xlab)



    def plot_mle_mm_lower_models(self, lower_order_df, parameters_dict):
        """Plot density and PP plots for models of lower order TEV distributions"""

        # TODO: this should be determined by the number of entries in the parameters dict
        n_row = 3
        n_col = 3

        fig1, ax1 = plt.subplots(n_row, n_col, figsize=(n_row*2, n_col*2), constrained_layout=True)
        fig2, ax2 = plt.subplots(n_row, n_col, figsize=(n_row*2, n_col*2), constrained_layout=True)


        hit_rank = 1

        for row in range(3):
            for col in range(3):

                cur_tevs = lower_order_df[lower_order_df['hit_rank'] == hit_rank]['tev']
                self.add_lower_model_plot(ax1[row, col], cur_tevs, parameters_dict, hit_rank)
                self.add_pp_plot(ax2[row, col], cur_tevs, parameters_dict, hit_rank)
                hit_rank += 1


        self.add_axis_labels(ax1, n_col, n_row, mode='density')
        self.add_axis_labels(ax2, n_col, n_row, mode='PP')

        fig1.savefig(f"./graphs/{self.out_name}_lower_models_rank{hit_rank}.png", dpi=600, bbox_inches="tight")
        fig2.savefig(f"./graphs/{self.out_name}_pp_plots_rank{hit_rank}.png", dpi=600, bbox_inches="tight")
        

    @staticmethod
    def add_pp_plot(axs, cur_tevs, parameters_dict, hit_rank):
        """Add P-P plot of the model with rank equal to "hit_rank" to plt.subplots"""

        mle_pars = parameters_dict['mle'][0].loc[:, hit_rank]
        mm_pars = parameters_dict['mm'][0].loc[:, hit_rank]

        mm_pp = of.TEVDistribution().cdf_asymptotic(cur_tevs, mm_pars[0], mm_pars[1], hit_rank)
        mle_pp = of.TEVDistribution().cdf_asymptotic(cur_tevs, mle_pars[0], mle_pars[1], hit_rank)
        emp_pp = np.arange(1, len(cur_tevs) + 1) / len(cur_tevs)

        axs.scatter(mle_pp, emp_pp, color='#D65215', s=1)
        axs.scatter(mm_pp, emp_pp, color='#2CB199', s=1)
        axs.plot([0,1], [0,1], color='k')


    @staticmethod
    def add_lower_model_plot(axs, cur_tevs, parameters_dict, hit_rank):
        """Plotting KDE for MLE and MM-based models of lower-order distributions"""

        def kde_plots(axes, kde_xs, parameters, order, color):
            mu, beta = parameters
            pdf_vals = of.TEVDistribution().pdf(kde_xs, mu, beta, order)
            axes.plot(kde_xs, pdf_vals, color=color)


        colors = ('#2D58B8', '#D65215', '#2CB199')
        kde_xs, kde_ys_observed = FFTKDE(bw=0.0005, kernel='gaussian').fit(cur_tevs).evaluate(2**8)

        if len(kde_ys_observed) == 0:
            return 0
        
        mle_pars = parameters_dict['mle'][0].loc[:, hit_rank]
        mm_pars = parameters_dict['mm'][0].loc[:, hit_rank]

        # plot the observed data KDE
        axs.plot(kde_xs, kde_ys_observed, color=next(colors))
        kde_plots(axs, kde_xs, mle_pars, hit_rank, color=next(colors))
        kde_plots(axs, kde_xs, mm_pars, hit_rank, color=next(colors))
        axs.set_ylim(0,)



    def plot_top_model_with_pi0(self, df, optimal_params):
        """find pi0 estimates for plotting the final models"""

        num_charges = len(optimal_params)
        colors = ('#2CB199', '#2CB199', '#D65215')
        fig, axs = plt.subplots(1, num_charges, figsize=(5 * num_charges, 5))


        for idx, charge in enumerate(optimal_params):

            top_tevs = df[(df['charge'] == charge) & (df['hit_rank'] == 1) ]['tev'].values
            mu, beta = optimal_params[charge]

            # get a rough approximation of pi0 for plotting only
            pi0 = len(top_tevs[top_tevs < 0.17]) / len(top_tevs)
            xs, kde_observed = FFTKDE(bw=0.01, kernel='gaussian').fit(top_tevs).evaluate(2**8)
            pdf_fitted = of.TEVDistribution().pdf(xs, mu, beta, order_index=0)

            axs[idx].fill_between(xs, kde_observed, alpha=0.2, color=colors[0], label='observed')
            axs[idx].plot(xs, kde_observed, color=colors[1])
            axs[idx].plot(xs,  pi0 * pdf_fitted, color=colors[2], linestyle='-', label='fitted')
            axs[idx].set_xlim(0.0, 0.6)
            axs[idx].set_ylim(0,)
            axs[idx].set_xlabel("TEV")
            axs[idx].set_ylabel("density")

        fig.tight_layout()
        fig.savefig(f"./{self.out_name}_fitted_top_models.png", dpi=600, bbox_inches="tight")

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

        fig.savefig(f"./{self.out_name}_lower_models_BIC_charge_{charge}.png", dpi=600, bbox_inches="tight")

    
    #### plotting for validation #####

    def __plot_boot_fdrs(self, axs, all_stats, pi_0):
        """plotting bootstrap FDP vs FDR"""

        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        for method in range(len(all_stats[0])):
            fdrs = all_stats[0][method][0][0,:]
            fdps = all_stats[0][method][1]
            if method == 0:
                self.__plot_fdr_stat(axs, pi_0*fdrs, np.array(fdps), cs_[method], xy_=1)
            else:
                self.__plot_fdr_stat(axs, pi_0*fdrs, np.array(fdps), cs_[method])

        axs.set_xlabel("FDR")
        axs.set_ylabel("FDP")


    def __plot_boot_tps(self, axs, all_stats):
        """plot boostrap # identified PSMs"""
        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        for method in range(len(all_stats[0])):
            fdrs = all_stats[0][method][0][0,:]
            tps = all_stats[0][method][2]
            if method == 0:
                self.__plot_fdr_stat(axs, fdrs, tps, cs_[method], axis_t="TPR")
            else:
                self.__plot_fdr_stat(axs, fdrs, tps, cs_[method], axis_t="TPR")

        axs.set_xlabel("FDR")
        axs.set_ylabel("Correctly identified PSMs")
        axs.set_ylim(0,17000)
        axs.set_xlim(0,0.1)


    @staticmethod
    def __plot_fdr_stat(axs, fdrs, fdp_stats, col, xy_=False, axis_t='FDP'):
        """plot FDP vs FDR - results of validation"""
        #fdrs = np.linspace(0.0001, 0.1, 100)

        if xy_:
            axs.plot([0.0001,0.1], [0.0001, 0.1], c='gray')

        axs.plot(fdrs, fdp_stats[0,:], color=col, linewidth=2)
        axs.fill_between(fdrs, fdp_stats[0,:], fdp_stats[2,:], alpha=0.2, color=col)
        axs.fill_between(fdrs, fdp_stats[0,:], fdp_stats[1,:], alpha=0.2, color=col)


        if axis_t == 'TPR':
            axs.set_xlim(-0.001, 0.1)
            axs.set_ylim(-0.01,)
        else:
            axs.set_xlim(-0.001, 0.1+0.001)
            axs.set_ylim(-0.001, 0.1+0.001)


