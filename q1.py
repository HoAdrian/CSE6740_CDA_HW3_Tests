OK_FORMAT = True

test = {   'name': 'q1',
    'points': None,
    'suites': [   {   'cases': [   {   'code': '>>> def _test_cnn_structure():\n'
                                               '...     import torch.nn as nn\n'
                                               '...     m = CNN((3, 32, 32), (16, 32), 10, 3, 2)\n'
                                               '...     kinds = [type(layer).__name__ for layer in m.features]\n'
                                               "...     assert kinds == ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d'] * 2, kinds\n"
                                               "...     assert isinstance(m.classifier, nn.Linear), 'classifier must be an nn.Linear'\n"
                                               '...     assert (m.classifier.in_features, m.classifier.out_features) == (32 * 8 * 8, 10)\n'
                                               '>>> _test_cnn_structure()\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 2},
                                   {   'code': '>>> def _test_cnn_forward():\n'
                                               '...     import torch\n'
                                               '...     torch.manual_seed(0)\n'
                                               '...     m = CNN((3, 32, 32), (16, 32), 10, 3, 2)\n'
                                               '...     logits = m(torch.randn(4, 3, 32, 32))\n'
                                               '...     assert logits.shape == (4, 10), logits.shape\n'
                                               '...     logits.sum().backward()\n'
                                               "...     assert m.features[0].weight.grad is not None, 'gradients must reach the first conv'\n"
                                               '>>> _test_cnn_forward()\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 2},
                                   {   'code': '>>> def _test_cnn_general():\n'
                                               '...     import torch\n'
                                               '...     m = CNN((1, 28, 28), (8, 16, 32), 7, 5, 2)\n'
                                               '...     convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]\n'
                                               '...     assert [tuple(c.weight.shape) for c in convs] == [(8, 1, 5, 5), (16, 8, 5, 5), (32, 16, 5, 5)]\n'
                                               '...     assert all((c.padding == (2, 2) for c in convs))\n'
                                               '...     assert m.classifier.in_features == 32 * 3 * 3, m.classifier.in_features\n'
                                               '...     assert m(torch.randn(2, 1, 28, 28)).shape == (2, 7)\n'
                                               '>>> _test_cnn_general()\n',
                                       'hidden': True,
                                       'locked': False,
                                       'points': 3}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
