import hilbert as h
from unittest import TestCase


class TestYieldsRecallable(TestCase):
    def test_yields_recallable(self):
        x_ = 5
        y_ = 4
        yy_ = 3
        xx_ = 2

        def f(x,y):
            return x,y

        g = h.factories.yields_recallable(f)
        f_, r = g(x=x_, y=y_)

        self.assertEqual(f_, (x_,y_))
        self.assertEqual(r(), (x_,y_))

        self.assertEqual(r(y=yy_), (x_,yy_))
        self.assertEqual(r(x=xx_), (xx_,y_))
        






