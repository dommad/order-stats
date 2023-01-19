"""Simulations to support assumptions and hypotheses made in the study"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import order_formulae as of
import importlib as imp
imp.reload(of)
ofs = of.Tools()


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class Simulator:
    """Conduct the simulations for the study"""

    def __init__(self) -> None:
        pass
    

    def simulate_pp_plots(self):
        """Compare finite N and asymptotic forms for k<=10 using P-P plot"""
        # no_cand = 100, k=10, samples = 1000
        mu_, beta = (0.02*np.log(1000), 0.02)
        sam = self.sample_tevs(20, 10, n_samples=1000, start_pars=(mu_, beta))
        fig, axs = plt.subplots(2,5, figsize=(12,5))
        
        for idx in range(10):
            fins = ofs.cdf_finite_n(sam[idx], mu_, beta, 10-idx, 100)
            asym = ofs.universal_cdf(sam[idx], mu_, beta, 10-idx)
            row, col = divmod(idx, 5)
            axs[row, col].scatter(fins, asym, c='#2D58B8', s=5)
            axs[row, col].plot([0,1], [0,1], c='grey')
            axs[row, col].set_xlabel("finite N CDF")
            axs[row, col].set_ylabel("asymptotic CDF")
            axs[row, col].set_title(f"order index k = {10-idx}")
        fig.tight_layout()
        fig.savefig("./pp_plot_simulation_test.png", dpi=600)


    @staticmethod
    def __tev_cdf(p_val, pars, num_points):
        """CDF value from TEV distribution"""
        mu_, beta = pars
        return mu_ - beta*np.log(num_points*(1-p_val))


    def sample_tevs(self, n_points, k_order, n_samples, start_pars):
        """Generate TEV distributions and sample top k_order statistics"""

        unif_sample = np.random.random((n_samples, n_points))
        tev_sample = self.__tev_cdf(unif_sample, start_pars, n_points)
        ordered = map(lambda x: sorted(x)[-k_order:], tev_sample)
        vals_grouped = list(zip(*ordered))

        return vals_grouped



    def estimate_params(self, dat, mode='finite'):
        """Estimate parameters for the provided ordered data,
        output comparison with starting params"""
        #mle_params = list(map(lambda x: lows.mle_mubeta(data[len(data)-x-1], x), range(len(data))))
        nop = len(dat)
        ran = range(nop)
        mu_, beta = (0.138, 0.02)
        if mode == 'finite':
            pars = list(map(lambda x: ofs.mle_fin_n(dat[nop-x-1], x, len(dat[nop-x-1])), ran))
        elif mode == 'asymptotic':
            pars = list(map(lambda x: ofs.mle_mubeta(dat[nop-x-1], x), ran))
        #mm_pars = list(map(lambda x: ofs.mm_estimator(dat[nop-x-1], x), dat_r))

        vals_fin = list(map(lambda x: ofs.cdf_finite_n(dat[nop-x-1], mu_, beta, x, 1000), ran))
        vals_asy = list(map(lambda x: ofs.universal_cdf(dat[nop-x-1], mu_, beta, x), ran))
        return pars, vals_fin, vals_asy


    def run_simulation(self, no_cand, no_k, mode_mle, mode_no_cand):
        """Run the simulation"""
        new_pars = []
        # colors = []
        new_vals_fin = []
        new_vals_asy = []

        i=0
        while i < 30:
            if mode_no_cand == 'random':
                no_cand = np.random.randint(100, 5000) # randomly selected

            ordered_tevs = self.sample_tevs(
                            no_cand,
                            no_k,
                            n_samples=1000,
                            start_pars=(0.13815510557964275, 0.02))

            pars, vals_f, vals_a = self.estimate_params(ordered_tevs, mode=mode_mle)
            new_pars += pars[4:] # start counting from top 5 downwards
            new_vals_fin.append(vals_f)
            new_vals_asy.append(vals_a)
            i +=1

        new_pars = np.array(new_pars)
        outname = f"{no_cand}_{no_k}_{mode_mle}_{mode_no_cand}"
        self.__plot_mubeta_scatter(new_pars, outname)
        return np.array(new_pars), np.array(new_vals_fin), np.array(new_vals_asy)

    @staticmethod
    def __plot_mubeta_scatter(data, outname):
        """Plots the scatterplot of mu vs beta estimates"""
        fig, axs = plt.subplots(figsize=(6,6))
        axs.scatter(data[:,0], data[:,1], s=5)
        axs.scatter([0.13815510557964275], [0.02], color='orange') # theoretical point

        l_r = st.linregress(data[:,0], data[:,1])
        x_s = np.array([min(data[:,0]), max(data[:,0])])
        y_s = l_r.slope*x_s + l_r.intercept
        axs.plot(x_s, y_s, color='k')
        print(l_r)

        axs.set_xlabel(r"$\mu$")
        axs.set_ylabel(r"$\beta$")

        #timestamp = datetime.now().strftime('%d-%m-%Y.%H_%M_%S')
        fig.savefig(f"{outname}_mubeta_scatter.png", dpi=400, bbox_inches='tight')


# simulation 1: draws from simulated TEV distribution, then test if asymptotic forms work on them
# aim: show that our analysis works on the perfect case OR reveal the estimator bias






# simulation 2: for draws from simulated TEV distribution, compare finite N vs asymptotic form
# aim: show that our analysis based on asymptotic forms is justified


# simulation 3: dependent RVs (how to do it?)
# aim: show that independence assumption is not necessary for our method to hold perfectly



# simulation 4: here put the code for Comet's e-value simulation and what's the problem with LR
