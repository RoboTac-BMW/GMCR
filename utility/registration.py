

class RegistrationResult:
    """
    Contains a registration result represented by scale, rotation and translation.
    """

    d_time_s = None
    d_time_R = None
    d_time_t = None

    true_inlier_mask = None
    start_pair_inlier_mask = None
    est_scale_inlier_mask = None
    est_rotation_inlier_mask = None
    est_translation_inlier_mask = None

    def __str__(self):
        line1 = f"s: {self.s}\n"
        line2 = f"R: {self.R}\n"
        line3 = f"t: {self.t}\n"
        return line1 + line2 + line3

    def __init__(self, s=None, R=None, t=None):

        self.s = s
        self.R = R
        self.t = t

    def get_inlier_percentages(self):

        res = {}

        if self.est_scale_inlier_mask is not None:
            s_inlier_perc = self.est_scale_inlier_mask.sum() / self.est_translation_inlier_mask.shape[0]
            res['s'] = s_inlier_perc

        if self.est_rotation_inlier_mask is not None:
            R_inlier_perc = self.est_rotation_inlier_mask.sum() / self.est_rotation_inlier_mask.shape[0]
            res['R'] = R_inlier_perc

        if self.est_translation_inlier_mask is not None:
            t_inlier_perc = self.est_translation_inlier_mask.sum() / self.est_translation_inlier_mask.shape[0]
            res['t'] = t_inlier_perc

        return res

    def get_inlier_retention_percentages(self):

        res = {}
        num_start_inlier_pairs = self.start_pair_inlier_mask.sum()

        if self.est_scale_inlier_mask is not None:
            s_retention_perc = self.est_scale_inlier_mask.sum() / num_start_inlier_pairs
            res['s'] = s_retention_perc

        if self.est_rotation_inlier_mask is not None:
            r_retention_perc = self.est_rotation_inlier_mask.sum() / num_start_inlier_pairs
            res['R'] = r_retention_perc

        if self.est_translation_inlier_mask is not None:
            t_retetntion_perc = self.est_translation_inlier_mask.sum() / num_start_inlier_pairs
            res['t'] = t_retetntion_perc

        return res
