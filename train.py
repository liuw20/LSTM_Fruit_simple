from Error import  pre_error
def train(epoch,model,train_x,train_y,loss_function,optimizer):
    out = model(train_x)
    loss = loss_function(out, train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+ 1) % 100 == 0:
        print('Epoch: {}, Loss:{:.5f}'.format((epoch + 1), loss.item()))
