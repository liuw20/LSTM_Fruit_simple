from torch import FloatTensor
from Error import pre_error


def predicate(model, test_x, test_y):
    test_x = test_x.type(FloatTensor)
    test_y = test_y.type(FloatTensor)
    model = model.eval()
    pre_test = model(test_x)
    pre_test = pre_test.view(-1).data.numpy()
    print('预测结果:{}\n'.format(pre_test))
    # print('实际结果:{}'.format(test_y))