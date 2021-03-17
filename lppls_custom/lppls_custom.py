from multiprocessing import Manager, Pool, Process
from matplotlib import pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import time

class LPPLS(object):

    def __init__(self, observations):
        """
        Args:
            observations (np.array,pd.DataFrame): 2xM matrix with timestamp and observed value.
        """
        assert isinstance(observations, (np.ndarray, pd.DataFrame)), \
            f'Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}'

        self.observations = observations
        self.coef_ = {}
        self.indicator_result_list = None

    @staticmethod
    @njit
    def lppls(t, tc, m, w, a, b, c1, c2):
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))

    def func_restricted(self, x, *args):
        """
        Finds the least square difference.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Args:
            x(np.ndarray):  1-D array with shape (n,).
            args:           Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """

        tc = x[0]
        m = x[1]
        w = x[2]
        obs = args[0]

        a, b, c1, c2 = self.matrix_equation(obs, tc, m, w)

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in obs[0, :]]
        delta = np.subtract(delta, obs[1, :])
        delta = np.power(delta, 2)

        return np.sum(delta)

    @staticmethod
    @njit
    def matrix_equation(observations, tc, m, w):
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        deltaT = tc - T
        phase = np.log(deltaT)
        fi = np.power(deltaT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)
        A = np.stack((np.ones_like(deltaT), fi, gi, hi))

        return np.linalg.lstsq(A.T, P, rcond=-1.0)[0]

    def fit(self, observations, max_searches, minimizer, bounds):
        """
        Args:
            observations (Mx2 numpy array): the observed time-series data.
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25.
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            bounds (list):
        Returns:
            tc, m, w, a, b, c, c1, c2
        """

        search_count = 0

        # find bubble
        while search_count < max_searches:
            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(bnd[0], bnd[1]) for bnd in bounds]
            #
            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]
            #
            seed = np.array([tc, m, w])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:
                tc, m, w, a, b, c, c1, c2 = self.minimize(observations, seed, minimizer, bounds)
                return tc, m, w, a, b, c, c1, c2
            except (np.linalg.LinAlgError, UnboundLocalError, ValueError):
                search_count += 1

        return 0, 0, 0, 0, 0, 0, 0, 0

    def minimize(self, observations, seed, minimizer, bounds):
        """
        Args:
            observations (np.ndarray):  the observed time-series data.
            seed (list):  time-critical, omega, and m.
            minimizer (str):  See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            bounds (list):
        Returns:
            tc, m, w, a, b, c, c1, c2
        """

        cofs = minimize(
            args=observations,
            fun=self.func_restricted,
            x0=seed,
            method=minimizer,
            bounds=bounds
        )

        if cofs.success:
            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]

            a, b, c1, c2 = self.matrix_equation(observations, tc, m, w)
            c = (c1 ** 2 + c2 ** 2) ** 0.5

            # Use sklearn format for storing fit params
            for coef in ['tc', 'm', 'w', 'a', 'b', 'c', 'c1', 'c2']:
                self.coef_[coef] = eval(coef)

            return tc, m, w, a, b, c, c1, c2
        else:
            raise UnboundLocalError

    def plot_fit(self):
        """
        Args:
            observations (Mx2 numpy array): the observed data
        Returns:
            nothing, should plot the fit
        """
        tc, m, w, a, b, c, c1, c2 = self.coef_.values()
        lppls_fit = self.lppls(self.observations[0], tc, m, w, a, b, c1, c2)

        data = pd.DataFrame({
            'Time': self.observations[0],
            'LPPLS Fit': lppls_fit,
            'Observations': self.observations[1],
        })
        data = data.set_index('Time')
        data.plot(figsize=(14, 8))

    def plot_confidence_indicators(self, result_index, title):
        """
        Args:
            result_index (int):
            title (str): super title for both subplots
        Returns:
            nothing, should plot the indicator
        """
        df = self.indicator_result_list[result_index]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 12))
        fig.suptitle(title)

        # plot pos bubbles
        ax1_0 = ax1.twinx()
        ax1.plot(df['pos_conf'].values, label='bubble indicator (pos)')
        ax1_0.plot(df['price'].values, color='orange')

        # plot neg bubbles
        ax2_0 = ax2.twinx()
        ax2.plot(df['neg_conf'].values, label='bubble indicator (neg)')
        ax2_0.plot(df['price'].values, color='orange')

        # set grids
        ax1.grid(which='major', axis='both', linestyle='--')
        ax2.grid(which='major', axis='both', linestyle='--')

        # set labels
        ax1.set_ylabel('bubble indicator (pos)')
        ax2.set_ylabel('bubble indicator (neg)')

        ax1_0.set_ylabel('price')
        ax2_0.set_ylabel('price')

        # ax1.legend(loc=2)
        # ax2.legend(loc=2)

    #
    def compute_indicator_process(
        self,
        window_size,
        smallest_window_size,
        filter_conditions_config,
        num_processes=1
    ):
        #*
        print('\n' + f'Now starting {num_processes} process(es)...')
        start = time.perf_counter()

        with Manager() as manager:
            self.indicator_result_list = manager.list()
            #
            processes = []

            for k in range(num_processes):
                #
                p = Process(
                      target=self._compute_indicator_pool,
                      args=[
                        window_size,
                        smallest_window_size,
                        filter_conditions_config,
                        self.indicator_result_list
                    ])
                p.start()
                processes.append(p)

            for process in processes:
                process.join()

        #*
        end = time.perf_counter()
        print(f'...finished in {round(end - start, 2)} second(s).' + '\n')

    #
    def _compute_indicator_pool(
        self,
        window_size,
        smallest_window_size,
        filter_conditions_config,
        indicator_result_list,
        increment=5,
        max_searches=25,
        workers=4
    ):
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0, :]) - window_size

        args_map = [(
            obs_copy[:, i:window_size + i],  # obs
            i,  # n_iter
            window_size,  # window_size
            smallest_window_size,  # smallest_window_size
            filter_conditions_config,
            increment,  # increment
            max_searches  # max_searches
        ) for i in range(obs_copy_len)]

        pool = Pool(processes=workers)

        result = pool.map(self._compute_indicator, args_map)
        pool.close()

        #
        indicator_result_list.append(
          self._res_to_df(
            result,
            list(filter_conditions_config[0].keys())[0]
        ))

    #
    def _compute_indicator(self, args):
        #
        obs, n_iter, window_size, smallest_window_size, filter_conditions_config, increment, max_searches = args
        #
        n_fits = (window_size - smallest_window_size) // increment
        #
        res = []

        # run n fits on the observation slice.
        for j in range(n_fits):
            obs_shrinking_slice = obs[:, j * increment:window_size + n_iter]

            #
            start = obs_shrinking_slice[0][0]
            end = obs_shrinking_slice[0][-1]
            delta = end - start

            # fit model to data and return params
            tc, m, w, a, b, c, c1, c2 = self.fit(
                obs_shrinking_slice,
                max_searches,
                minimizer='SLSQP', # Sequential Least SQuares Programming
                bounds=[ # search space guidance from Shu & Zhu (2019)
                  (end, end + (delta / 3)), # tc
                  (0, 1), # m
                  (1, 50) # w
                ]
            )

            qualified = {}

            # TODO: add docstring
            # filter_conditions_config = [
            #   {'condition_1':[tc_range, m_range, w_range, O_min, D_min]},
            #   {'condition_2':[tc_range, m_range, w_range, O_min, D_min]}
            # ]

            for condition in filter_conditions_config:
                for value in condition:
                    tc_min, tc_max = eval(condition[value][0])
                    m_min, m_max = condition[value][1]
                    w_min, w_max = condition[value][2]
                    O_min = condition[value][3]
                    D_min = condition[value][4]

                    tc_in_range = tc_min < tc < tc_max
                    m_in_range = m_min < m < m_max
                    w_in_range = w_min < w < w_max

                    ## formulas from Shu & Zhu (2019)
                    # number of Oscillations (half-periods), used to distinguish a genuine log-periodic signal from noise
                    O_in_range = (w / 2) * np.log((tc - start) / (tc - end)) > O_min
                    # Damping parameter expresses the crash hazard rate h(t) is non-negative by definition
                    D_in_range = (m * abs(b)) / (w * c) > D_min

                    if tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range:
                        is_qualified = True
                    else:
                        is_qualified = False

                    qualified[value] = is_qualified

            sign = 1 if b < 0 else -1

            res.append({
                'tc': tc,
                'm': m,
                'w': w,
                'a': a,
                'b': b,
                'c': c,
                'c1': c1,
                'c2': c2,
                'qualified': qualified,
                'sign': sign,
                't1': start,
                't2': end
            })

        return res

    def _res_to_df(self, res, condition_name):
        """
        Args:
            res (list): result from mp_compute_indicator
            condition_name (str): the name you assigned to the filter condition in your config
        Returns:
            pd.DataFrame()
        """
        idx = self.observations[0, :]
        price = self.observations[1, :]
        n = len(price) - len(res)
        # pos_fits = [0] * n
        # neg_fits = [0] * n
        pos_conf_lst = [0] * n
        neg_conf_lst = [0] * n
        fits_ = [0] * n

        for r in res:
            pos_count = 0
            neg_count = 0
            pos_true_count = 0
            neg_true_count = 0
            for fits in r:

                if fits['sign'] > 0:
                    pos_count += 1
                    if fits['qualified'][condition_name]:
                        pos_true_count += 1
                if fits['sign'] < 0:
                    neg_count += 1
                    if fits['qualified'][condition_name]:
                        neg_true_count = neg_true_count + 1
            # pos_conf_lst.append(pos_true_count / len(r))
            # neg_conf_lst.append(neg_true_count / len(r))
            fits_.append(fits)
            pos_conf_lst.append(pos_true_count / pos_count if pos_count > 0 else 0)
            neg_conf_lst.append(neg_true_count / neg_count if neg_count > 0 else 0)

        return pd.DataFrame({
            'idx': idx,
            'price': price,
            'pos_conf': pos_conf_lst,
            'neg_conf': neg_conf_lst,
            'fit_params': fits_,
        }).set_index('idx')
