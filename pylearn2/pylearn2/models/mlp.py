#
#  Add this node to the original
#  /pylearn2/pylearn2/mlp.py to '
#  get RectifiedLog activation function
#
#########################################
##Experimental Node
########################################
class RectifiedLog(Linear):

    def __init__(self, left_slope=0.0, **kwargs):
        super(RectifiedLog, self).__init__(**kwargs)
        self.left_slope = left_slope

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        # Original: p = p * (p > 0.) + self.left_slope * p * (p < 0.)
        # T.switch is faster.
        # For details, see benchmarks in
        # pylearn2/scripts/benchmark/time_relu.py
        p = T.switch(p > 0., T.log(1.+p), self.left_slope * p)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()
#######################################################################
#######################################################################