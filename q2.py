OK_FORMAT = True

test = {   'name': 'q2',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> def _test_reparameterize():\n'
                                               '...     vae = VAE(image_size=4, z_dim=3)\n'
                                               '...     mu = torch.tensor([[0.0, 1.0, -1.0], [2.0, -0.5, 0.3]], requires_grad=True)\n'
                                               '...     log_var = torch.tensor([[0.0, 2.0, -2.0], [1.0, 0.0, -4.0]], requires_grad=True)\n'
                                               '...     torch.manual_seed(0)\n'
                                               '...     z = vae.reparameterize(mu, log_var)\n'
                                               '...     torch.manual_seed(0)\n'
                                               '...     z_expected = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)\n'
                                               '...     assert z.shape == mu.shape, z.shape\n'
                                               '...     assert torch.allclose(z, z_expected)\n'
                                               '...     z.sum().backward()\n'
                                               "...     assert mu.grad is not None and log_var.grad is not None, 'z must be differentiable w.r.t. mu and log_var'\n"
                                               '>>> _test_reparameterize()\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 8}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
