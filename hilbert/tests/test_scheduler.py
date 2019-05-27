from unittest import TestCase
import hilbert as h
import torch


def is_close(a, b, tol=1e-6):
    if abs(a-b) < tol:
        return True
    return False


class TestSchedulers(TestCase):

    def test_temp_scheduler(self):

        init_temp = 2
        milestone_sets = [0, 10, 14], [5, 10, 14]
        temperatures = [3, 2, 1]

        for milestones in milestone_sets:

            loss = h.loss.MLELoss(100, 2)

            # The temperature is what we passed to the loss function
            self.assertEqual(loss.temperature, init_temp)

            temp_scheduler = h.scheduler.TempScheduler(
                loss, milestones, temperatures)

            # If there's milestone at 0, immediately apply its temp.
            if milestones[0] == 0:
                self.assertEqual(loss.temperature, temperatures[0])

            epoch = 0
            # Temperature gets updated as expected
            for milestone_id in range(len(milestones)):
                while epoch < milestones[milestone_id]:
                    if milestone_id == 0:
                        self.assertEqual(loss.temperature, init_temp)
                    else:
                        self.assertEqual(
                            loss.temperature, temperatures[milestone_id-1])
                    temp_scheduler.step()
                    epoch += 1

            # After reaching the final milestone, the final temp is applied
            self.assertEqual(loss.temperature, temperatures[-1])

            # After applying final temp, no more changes to temp occur.
            for i in range(10):
                temp_scheduler.step()
            self.assertEqual(loss.temperature, temperatures[-1])




    def test_linear_lr_scheduler(self):

        start_lr = 10
        num_epochs = 20
        end_lr = 1

        params1 = (
            torch.nn.Parameter(torch.rand(5)),
            torch.nn.Parameter(torch.rand(5))
        )
        params2 = (
            torch.nn.Parameter(torch.rand(5)),
            torch.nn.Parameter(torch.rand(5))
        )
        opt = torch.optim.SGD(
            [{'params':params1, 'lr':5},{'params':params2}], lr=3)

        # initially the learning rates are as provided
        self.assertEqual(opt.param_groups[0]['lr'], 5)
        self.assertEqual(opt.param_groups[1]['lr'], 3)

        scheduler = h.scheduler.LinearLRScheduler(
            opt, start_lr, num_epochs, end_lr)

        # immediately after creating the scheduler, it sets the learning rates
        self.assertEqual(opt.param_groups[0]['lr'], 10)
        self.assertEqual(opt.param_groups[1]['lr'], 10)

        # Learning rates are updated as expected
        for epoch in range(30):
            if epoch < 20:
                expected_lr = start_lr-(start_lr-end_lr)*(epoch/num_epochs)
            else:
                expected_lr = end_lr
            for param_group in opt.param_groups:
                self.assertTrue(is_close(param_group['lr'], expected_lr))
            scheduler.step()








