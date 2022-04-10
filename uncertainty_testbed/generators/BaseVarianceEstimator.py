import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
from loop_helper import safe_fit
from config import BOOTSTRAP, GBM, RF, LOGISTIC, SINGLE_TREE, BETA_TREE

__author__ = "nikos.daniilidis"


class VarianceEstimator:
    """
    Estimator for the variances or standard deviations of the predictions of a set of models
    """
    def __init__(self, kind):
        """
        :param kind: string. What type of variance estimator to build
        :return: Nothing
        """
        assert kind in [BOOTSTRAP, BETA_TREE]  # TODO: add "score density", "random forest",
        self.kind = kind
        self.ensembles = None  # list of list of classifier for "bootstrap" variance estimation
        self.betaclfs = None  # list(tree classifier) used to make predictions from a Beta estimator using a single tree
        self.tds = None  # list(dictionary) used to make predictions from a Beta estimator using a single tree

    def predict_var(self, x):
        """
        Return the variance of the model predictions for each event in x
        :param x: numpy array of shape [n_events, n_features]
        :return: numpy array of shape [n_events, n_streams]
        """
        if x is None:  # when called with x == None, can check if returning variance or sd
            return "variance estimator"
        return np.power(self.predict_sd(x), 2)

    def predict_sd(self, x):
        """
        Return the standard deviation of the model predictions for each event in x
        :param x: numpy array of shape [n_events, n_features]
        :return: numpy array of shape [n_events, n_streams]
        """
        if x is None:  # when called with x == None, can check if returning variance or sd
            return "standard deviation estimator"
        if self.kind == BOOTSTRAP:
            return self.__boot_sd(x)
        elif self.kind == BETA_TREE:
            return self.__beta_tree_sd(x)
        else:
            warnings.warn("kind = %s is not implemented in VarianceEstimator" % self.kind)

    def __boot_sd(self, x):
        """
        Return the bootstrap standard deviation of the model predictions for each event in x
        :param x: numpy array of shape [n_events, n_features]
        :return: numpy array of shape [n_events, n_streams]
        """
        for ix in range(len(self.ensembles)):
            assert len(self.ensembles[0]) == len(self.ensembles[ix])
        num_events = x.shape[0]
        num_streams = len(self.ensembles)
        num_boots = len(self.ensembles[0])
        p_ensembles = [np.zeros((num_events, num_boots))] * num_streams
        p_sd = [np.zeros((num_events, 1))] * num_streams
        for stream in range(num_streams):
            for im, model in enumerate(self.ensembles[stream]):
                p_ensembles[stream][:, im] = model.predict_proba(x)[:, 1].T
            p_sd[stream] = np.std(p_ensembles[stream], axis=1)
        return np.vstack(p_sd).T

    def __beta_tree_sd(self, x):
        """
        Return an estimate of the standard deviation of any classifier will have on predictions on a data set.
        The estimate comes from using a tree trained on the training data to estimate Beta distributions for the samples
        from x which fall on the leaves of the trees.
        :param x: numpy array of shape [n_events, n_features]. The data set on which to estimate standard deviation
        :return: numpy array of shape [n_events, n_streams]
        """
        for betaclf in self.betaclfs:
            assert isinstance(betaclf, sklearn.tree.tree.DecisionTreeClassifier)
        num_events = x.shape[0]
        num_streams = len(self.betaclfs)
        p_sd = [np.zeros((num_events, 1))] * num_streams
        for stream in range(num_streams):
            betaclf, td = self.betaclfs[stream], self.tds[stream]
            sdd = {ky: item["sd"] for ky, item in td.iteritems()}
            b = betaclf.apply(x)
            sd = np.vectorize(sdd.get, otypes=[np.float])
            p_sd[stream] = sd(b)

        return np.vstack(p_sd).T  # TODO: Check that this returns the correct shape

    def train(self, xs, ys, ms, **kwargs):
        """
        Train a variance estimator
        :param xs: list of numpy array. The features data
        :param ys: list of numpy array. The labels data
        :param ms: list of sklearn classifiers or None. If a classifier is None, predictions will be random
        :param kwargs:
        :return: list of list of classifier. Outer list runs through streams. Inner lists are n_boots classifiers for
                each stream
        """
        assert len(xs) == len(ys)
        if self.kind == BOOTSTRAP:
            self.ensembles = self.__boot_train(xs, ys, ms, **kwargs)
            return self.ensembles
        elif self.kind == BETA_TREE:
            self.betaclfs, self.tds = self.__tree_train(xs, ys, **kwargs)
            return None
        else:
            warnings.warn("kind = %s is not implemented in VarianceEstimator" % self.kind)

    def __boot_train(self, xs, ys, ms, n_boots=10, predictor=None, replace=False, params=None):
        """
        Return an ensemble of n_boots classifiers for each stream. For each stream the ensemble consists of classifiers
        each trained on a 1-1/n_boots fraction of the data if replace is False, else on a sample with the same size as
        the data.
        :param xs: list of numpy array. The features data
        :param ys: list of numpy array. The labels data
        :param ms: list of sklearn classifiers or None. If a classifier is None, predictions will be random
        :param n_boots: integer. The number of bootstrap classifiers
        :param predictor: string. Type of classifier to train
        :param replace: boolean. Sample with replacement for the bootstrap
        :param params: dict. Parameters to use for the fitted classiifers.
        :return: list of list of classifier. Outer list runs through streams. Inner lists are n_boots classifiers for
                each stream
        """
        # TODO: When updating predictor to list of string, also update params to list of dict
        assert len(xs) == len(ms)
        ensembles = [[] for m in ms]  # list of lists to hold the n_boots models for each stream
        for ix, x in enumerate(xs):
            if predictor is None:
                try:
                    predictor = self.__predictor_mapper(ms[ix].estimator)
                except NotImplementedError:
                    warnings.warn(
                        'Encountered unknown predictor type at VarianceEstimator.__boot_train. Aborting bootstrap.')
                    ensembles[ix] = [None for m in ms]
                    break
            for b in range(n_boots):
                y, l = ys[ix], len(ys[ix])
                if replace:
                    n_tr = l
                else:
                    n_tr = int(l*(n_boots-1.)/n_boots)
                indices = np.random.choice(np.arange(l), size=n_tr, replace=replace)
                xt, yt = x[indices, :], y[indices]
                m = ms[ix]
                if params is None:
                    ps = m.best_params_
                else:
                    ps = params
                if predictor == GBM:
                    learning_rate, n_estimators, max_depth = ps['learning_rate'], ps['n_estimators'], ps['max_depth']
                    clf = GradientBoostingClassifier(
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        max_depth=max_depth)
                elif predictor == RF:
                    n_estimators, max_depth = ps['n_estimators'], ps['max_depth']
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth)
                elif predictor == LOGISTIC:
                    c = ps['C']
                    clf = LogisticRegression(C=c)
                elif predictor == SINGLE_TREE:
                    max_depth, min_samples_split, min_samples_leaf = ps["max_depth"], ps["min_samples_split"], \
                                                                     ps["min_samples_leaf"]
                    clf = DecisionTreeClassifier(max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf)
                else:
                    warnings.warn("predictor = %s is not implemented in VarianceEstimator\n" % predictor)
                    return
                clf = safe_fit(clf, xt, yt, safe_min=1)  # fit adding dummy elements of minority class if too few
                ensembles[ix].append(clf)
        return ensembles

    def __tree_train(self, xs, ys, aprior=0.1, bprior=1.,
                     params=(('min_samples_split', 40), ('min_samples_leaf', 20), ('max_depth', [6, 7, 8, 9, 10]),
                             ('folds', 50), ('n_jobs', 2)), predictor=None, replace=False):
        """
        Train a number of trees on the data sets xs and ys. Each tree maps samples to leaves of the tree. Each leaf of
        the tree contains a number of 0s and 1s from the train data. These are used to update a Beta estimate of the
        probability of a 1 on the leaf of the tree.
        :param xs: list of numpy array. The features data
        :param ys: list of numpy array. The labels data
        :param aprior: float. Prior of the alpha parameter of the beta distributions
        :param bprior: float. Prior of the beta parameter of the beta distributions
        :param params: List of tuple. Parameters to fit the trees
        :return: list of trained tree classifiers and list of dictionaries with the Beta parameters for each leaf
                of each tree
        """
        if predictor is not None or replace is not None:
            warnings.warn("Beta tree trainer will ignore unexpected arguments")
            warnings.warn("predictor = %s\nreplace = %s" % (str(predictor), str(replace)))
        paramsd = dict(params)
        betaclfs, tds = [], []
        min_samples_split, min_samples_leaf = paramsd['min_samples_split'], paramsd['min_samples_leaf']
        max_depth, folds, n_jobs = paramsd['max_depth'], paramsd['folds'], paramsd['n_jobs']
        param_grid = {'max_depth': max_depth}
        for ix, x in enumerate(xs):
            y = ys[ix]
            clf = GridSearchCV(DecisionTreeClassifier(criterion='gini', splitter='best',
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf),
                               param_grid=param_grid, cv=folds, n_jobs=n_jobs)
            betaclf = safe_fit(clf, x, y, safe_min=folds)
            betaclfs.append(betaclf.best_estimator_)
            tds.append(self.__tree_dict(betaclf.best_estimator_, x, y, aprior=aprior, bprior=bprior))
        return betaclfs, tds

    @staticmethod
    def __bucket_count_tuples(clf, x):
        # TODO: Docstring
        assert isinstance(clf, sklearn.tree.tree.DecisionTreeClassifier)
        buckets = clf.apply(x)
        counts = np.bincount(buckets)
        bb = np.nonzero(counts)[0]
        return zip(bb, counts[bb])

    def __tree_dict(self, clf, x, y, aprior, bprior):
        # TODO: Docstring
        assert isinstance(clf, sklearn.tree.tree.DecisionTreeClassifier)
        tuples = self.__bucket_count_tuples(clf, x)
        atuples = self.__bucket_count_tuples(clf, x[y == 1])
        btuples = self.__bucket_count_tuples(clf, x[y == 0])

        nd = dict(tuples)
        ad = dict(atuples)
        bd = dict(btuples)
        d = {}
        for ky in nd.keys():  # TODO: this is ugly, but ok
            n = nd[ky]
            if ky in ad.keys():
                alpha = aprior + ad[ky]
            else:
                alpha = aprior
            if ky in bd.keys():
                beta = bprior + bd[ky]
            else:
                beta = bprior
            d[ky] = {}
            d[ky]["n"] = n
            d[ky]["alpha"] = alpha
            d[ky]["beta"] = beta
            d[ky]["variance"] = alpha*beta / ((alpha+beta)**2 * (alpha+beta+1))
            d[ky]["sd"] = np.sqrt( alpha*beta / ((alpha+beta)**2 * (alpha+beta+1)) )

        return d

    @staticmethod
    def __predictor_mapper(model):
        """
        Map models to model type aliases.
        :param model: sklearn model
        :return: string. One of the implemented model classes.
        """
        if isinstance(model, sklearn.ensemble.gradient_boosting.GradientBoostingClassifier):
            return GBM
        elif isinstance(model, sklearn.ensemble.RandomForestClassifier):
            return RF
        elif isinstance(model, sklearn.linear_model.LogisticRegression):
            return LOGISTIC
        elif isinstance(model, sklearn.linear_model.LogisticRegressionCV):
            return LOGISTIC
        else:
            raise NotImplementedError

# TODO: Tests for VarianceEstimator