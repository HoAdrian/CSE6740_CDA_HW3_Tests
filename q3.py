OK_FORMAT = True

test = {   'name': 'q3',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> def _test_reconstr_losses():\n'
                                               '...     x = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]])\n'
                                               '...     x_hat = torch.tensor([[0.1, 0.9, 0.8, 0.2], [0.7, 0.3, 0.6, 0.9]])\n'
                                               '...     rec = reconst_loss(x_hat, x)\n'
                                               "...     rec_expected = F.binary_cross_entropy(x_hat, x, reduction='sum')\n"
                                               "...     assert rec.ndim == 0, 'reconst_loss must return a scalar'\n"
                                               "...     assert torch.allclose(rec, rec_expected), 'use BCE summed over the batch, not the mean'\n"
                                               '>>> _test_reconstr_losses()\n'
                                               '/opt/miniconda3-pbe/envs/cs6740-hw3/lib/python3.11/site-packages/torch/nn/_reduction.py:60: UserWarning: size_average and reduce args will be '
                                               "deprecated, please use reduction='sum' instead.\n"
                                               '  return get_enum(legacy_get_string(size_average, reduce, emit_warning))\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 3},
                                   {   'code': '>>> def _test_kl():\n'
                                               '...     mu = torch.tensor([[0.0, 0.0], [1.0, -0.5]])\n'
                                               '...     log_var = torch.tensor([[0.0, 0.0], [0.5, -1.0]])\n'
                                               '...     kl = kl_div(log_var, mu)\n'
                                               '...     kl_expected = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())\n'
                                               "...     assert kl.ndim == 0, 'kl_div must return a scalar'\n"
                                               '...     assert torch.allclose(kl, kl_expected)\n'
                                               "...     assert torch.isclose(kl_div(torch.zeros(2, 3), torch.zeros(2, 3)), torch.tensor(0.0)), 'KL(N(0,I) || N(0,I)) is 0'\n"
                                               '>>> _test_kl()\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 4}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
