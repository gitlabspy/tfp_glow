"""
ref : https://github.com/tensorflow/probability/issues/1006
kwargs = make_bijector_kwargs(transformedDistribution.bijector, {'glowblock_2_rnvp' : {'conditional_inputs' : condition} })
transformedDistribution.log_prob(x, **{'bijector_kwargs' : kwargs })
"""
def make_bijector_kwargs(self, bijector, name_to_kwargs):
    if hasattr(bijector, 'bijectors'):
        return {b.name: self.make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if name_regex in bijector.name:
                return kwargs
    return {}
